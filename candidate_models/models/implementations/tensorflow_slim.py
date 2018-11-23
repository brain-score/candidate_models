import glob
import logging
import os
from collections import OrderedDict

import networkx as nx
import pandas as pd
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
        self._placeholder, model_inputs = self._create_inputs(batch_size, image_size)
        self._logits, self.endpoints = self._create_model(model_inputs)
        self._sess = tf.Session()
        self._restore(weights)

    def _create_inputs(self, batch_size, image_size):
        raise NotImplementedError()

    def _create_model(self, inputs):
        raise NotImplementedError()

    def _restore(self, weights):
        raise NotImplementedError()

    def _load_images(self, image_filepaths, image_size):
        return image_filepaths

    def _load_image(self, image_filepath):
        image = tf.read_file(image_filepath)
        image = tf.image.decode_png(image, channels=3)
        return image

    def _preprocess_images(self, images, image_size):
        return images

    def _preprocess_image(self, image, image_size):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, (image_size, image_size))
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        return image

    def _get_activations(self, images, layer_names):
        layer_tensors = OrderedDict((layer, self.endpoints[layer]) for layer in layer_names)
        layer_outputs = self._sess.run(layer_tensors, feed_dict={self._placeholder: images})
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
        placeholder = tf.placeholder(dtype=tf.string, shape=[batch_size])

        process_imagepath = lambda image_path: self._preprocess_image(self._load_image(image_path), image_size)

        # we're not using preprocess_factory because we need to restrict inception preprocessing from cropping
        if model_properties['preprocessing'] == 'vgg':
            preprocess_image = lambda image: vgg_preprocessing.preprocess_image(
                image, image_size, image_size, resize_side_min=image_size)
        elif model_properties['preprocessing'] == 'inception':
            preprocess_image = lambda image: inception_preprocessing.preprocess_for_eval(
                image, image_size, image_size, central_fraction=1.)
        else:
            raise ValueError(f"unknown preprocessing {model_properties['preprocessing']}")
        preprocess = lambda image_path: preprocess_image(process_imagepath(image_path))

        preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)
        return placeholder, preprocess

    def _create_model(self, inputs):
        from nets import nets_factory
        model_properties = self._get_model_properties(self._model_name)
        model_identifier = model_properties['callable']
        if model_identifier.startswith('mobilenet'):  # strip image size from mobilenet name
            model_identifier = '_'.join(model_identifier.split('_')[:-1])
        model = nets_factory.get_network_fn(model_identifier,
                                            num_classes=1001 - int(model_properties['labels_offset']),
                                            is_training=False)
        return model(inputs)

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
            # Restore using exponential moving average since it produces (1.5-2%) higher accuracy according to
            # https://github.com/tensorflow/models/blob/a6494752575fad4d95e92698dbfb88eb086d8526/research/slim/nets/mobilenet/mobilenet_example.ipynb
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
