import glob
import logging
import os
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
import tensorflow as tf

from candidate_models import s3
from candidate_models.models.implementations import DeepModel, Defaults

_logger = logging.getLogger(__name__)

slim_models = pd.read_csv(os.path.join(os.path.dirname(__file__), 'models.csv'))
slim_models = slim_models[slim_models['framework'] == 'slim']


class TensorflowSlimModel(DeepModel):
    def __init__(self, weights=Defaults.weights,
                 batch_size=Defaults.batch_size, image_size=Defaults.image_size):
        super().__init__(batch_size=batch_size, image_size=image_size)
        self.inputs = self._create_inputs(batch_size, image_size)
        self._logits, self.endpoints = self._create_model(self.inputs)
        self._sess = tf.Session()
        self._restore(weights)

    def _create_inputs(self, batch_size, image_size):
        raise NotImplementedError()

    def _create_model(self, inputs):
        raise NotImplementedError()

    def _restore(self, weights):
        raise NotImplementedError()

    def _load_image(self, image_filepath):
        image = skimage.io.imread(image_filepath)
        return image

    def _preprocess_images(self, images, image_size):
        images = [self._preprocess_image(image, image_size) for image in images]
        return np.array(images)

    def _preprocess_image(self, image, image_size):
        image = skimage.transform.resize(image, (image_size, image_size))
        assert image.min() >= 0
        assert image.max() <= 1
        if image.ndim == 2:  # binary
            image = skimage.color.gray2rgb(image)
        assert image.ndim == 3
        return image

    def _get_activations(self, images, layer_names):
        layer_tensors = OrderedDict((layer, self.endpoints[layer]) for layer in layer_names)
        layer_outputs = self._sess.run(layer_tensors, feed_dict={self.inputs: images})
        return layer_outputs

    def graph(self):
        g = nx.DiGraph()
        for name, layer in self.endpoints.items():
            g.add_node(name, object=layer, type=type(layer))
        g.add_node("logits", object=self._logits, type=type(self._logits))
        return g


class TensorflowSlimPredefinedModel(TensorflowSlimModel):
    def __init__(self, model_name, *args, **kwargs):
        self._model_name = model_name
        tf.reset_default_graph()
        super().__init__(*args, **kwargs)

    def _create_inputs(self, batch_size, image_size):
        from preprocessing import inception_preprocessing, vgg_preprocessing
        model_properties = self._get_model_properties(self._model_name)
        inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
        preprocess_image = vgg_preprocessing.preprocess_image if model_properties['preprocessing'] == 'vgg' \
            else inception_preprocessing.preprocess_image
        return tf.map_fn(lambda image: preprocess_image(tf.image.convert_image_dtype(image, dtype=tf.uint8),
                                                        image_size, image_size), inputs)

    def _create_model(self, inputs):
        from nets import nets_factory
        model_properties = self._get_model_properties(self._model_name)
        call = model_properties['callable']
        arg_scope = nets_factory.arg_scopes_map[call](weight_decay=0.)
        kwargs = {}
        if self._model_name.startswith('mobilenet_v2') or self._model_name.startswith('mobilenet_v1'):
            arg_scope = nets_factory.arg_scopes_map[call](weight_decay=0., is_training=False)
            kwargs = {'depth_multiplier': model_properties['depth_multiplier']}
        model = nets_factory.networks_map[call]
        with tf.contrib.slim.arg_scope(arg_scope):
            logits, endpoints = model(inputs,
                                      num_classes=1001 - int(model_properties['labels_offset']),
                                      is_training=False,
                                      **kwargs)
            return logits, endpoints

    def _get_model_properties(self, model_name):
        _model_properties = slim_models[slim_models['model'] == model_name]
        _model_properties = {field: next(iter(_model_properties[field]))
                             for field in _model_properties.columns}
        return _model_properties

    def _restore(self, weights):
        if weights is None:
            return
        assert weights == 'imagenet'
        var_list = None
        if self._model_name.startswith('mobilenet'):
            # Restore using exponential moving average since it produces (1.5-2%) higher accuracy
            ema = tf.train.ExponentialMovingAverage(0.999)
            var_list = ema.variables_to_restore()
        restorer = tf.train.Saver(var_list)

        restore_path = self._find_model_weights(self._model_name)
        restorer.restore(self._sess, restore_path)

    def _find_model_weights(self, model_name):
        framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
        weights_path = os.getenv('CM_TSLIM_WEIGHTS_DIR', os.path.join(framework_home, 'model-weights', 'slim'))
        model_path = os.path.join(weights_path, model_name)
        if not os.path.isdir(model_path):
            self._logger.debug(f"Downloading weights for {model_name} to {model_path}")
            os.makedirs(model_path)
            s3.download_folder(f"slim/{model_name}", model_path)
        fnames = glob.glob(os.path.join(model_path, '*.ckpt*'))
        assert len(fnames) > 0
        restore_path = fnames[0].split('.ckpt')[0] + '.ckpt'
        return restore_path
