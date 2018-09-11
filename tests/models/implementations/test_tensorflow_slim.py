import logging
import os
import shutil

import numpy as np
import pytest
from tests.models import mock_imagenet

from brainscore.assemblies import NeuroidAssembly
from candidate_models.models.implementations.tensorflow_slim import TensorflowSlimModel, TensorflowSlimPredefinedModel


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
