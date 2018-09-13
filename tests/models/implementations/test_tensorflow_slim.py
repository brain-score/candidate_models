import logging
import os
import shutil

import numpy as np
import pytest
import tensorflow as tf
from preprocessing import vgg_preprocessing
from tests.models import mock_imagenet

from brainscore.assemblies import NeuroidAssembly
from candidate_models.models.implementations.tensorflow_slim import TensorflowSlimModel, TensorflowSlimPredefinedModel

slim = tf.contrib.slim


class TestLoadPreprocessImage:
    images = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg']

    @pytest.mark.parametrize('filename', images)
    def test_rgb(self, filename):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        img = TensorflowSlimModel._load_image(None, filepath)
        img = TensorflowSlimModel._preprocess_image(None, img, image_size=224)
        np.testing.assert_array_equal(img.shape, [224, 224, 3])


class TestModels:
    def test_single_image_resnet101v2(self):
        model = TensorflowSlimPredefinedModel('resnet-101_v2', batch_size=1, image_size=224)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=None)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 1

    def test_single_image_pad_resnet101v2(self):
        model = TensorflowSlimPredefinedModel('resnet-101_v2', batch_size=2, image_size=224)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=None)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 1

    def test_two_batches_single_image_resnet101v2(self):
        model = TensorflowSlimPredefinedModel('resnet-101_v2', batch_size=1, image_size=224)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')] * 2
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=None)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 2

    def test_two_batches_two_images_resnet101v2(self):
        model = TensorflowSlimPredefinedModel('resnet-101_v2', batch_size=2, image_size=224)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')] * 4
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=None)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 4

    def test_pca_resnet101v2(self):
        model = TensorflowSlimPredefinedModel('resnet-101_v2', batch_size=1, image_size=224)
        mock_imagenet(model)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=10)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 1
        assert len(activations['neuroid_id']) == 10

    def test_pca_small_layer(self):
        class SmallLayerModel(TensorflowSlimModel):
            def _create_inputs(self, batch_size, image_size):
                inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
                preprocess_image = vgg_preprocessing.preprocess_image
                return tf.map_fn(lambda image: preprocess_image(tf.image.convert_image_dtype(image, dtype=tf.uint8),
                                                                image_size, image_size), inputs)

            def _create_model(self, inputs):
                with tf.variable_scope('my_model', values=[inputs]) as sc:
                    end_points_collection = sc.original_name_scope + '_end_points'
                    # Collect outputs for conv2d, fully_connected and max_pool2d.
                    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                        outputs_collections=[end_points_collection]):
                        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
                        net = slim.max_pool2d(net, [5, 5], 5, scope='pool1')
                        net = slim.max_pool2d(net, [3, 3], 5, scope='pool2')
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                        return net, end_points

            def _restore(self, weights):
                assert weights is None
                init = tf.initialize_all_variables()
                self._sess.run(init)

        model = SmallLayerModel(weights=None, batch_size=1, image_size=224)
        mock_imagenet(model)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]
        activations = model.get_activations(stimuli_paths, layers=['my_model/pool2'], pca_components=300)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 1
        assert len(activations['neuroid_id']) == 256  # i.e. not 300 but the smaller layer number


class TestAutomaticWeights:
    def test_download(self):
        model = 'mobilenet_v1_0.25_128'
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model-weights', 'slim', model)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)  # ensure weights haven't been downloaded yet

        class SelfMock(object):
            def __init__(self):
                self._logger = logging.getLogger(TensorflowSlimPredefinedModel.__name__)

        TensorflowSlimPredefinedModel._find_model_weights(self=SelfMock(), model_name=model)
        assert len(os.listdir(model_path)) == 7


class TestGraph:
    def test_resnet101(self):
        model = TensorflowSlimPredefinedModel('resnet-101_v2')
        graph = model.graph()
        assert 146 == len(graph.nodes)
        assert 101 == len([node_name for node_name, node in graph.nodes.items()
                           if 'conv' in node['object'].name.lower() or 'logits' in node['object'].name.lower()])
