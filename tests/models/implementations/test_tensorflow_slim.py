import os

import numpy as np

from brainscore.assemblies import NeuroidAssembly
from candidate_models.models.implementations.tensorflow_slim import TensorflowSlimModel
from tests.models import mock_imagenet
from tests.models.implementations import get_grayscale_image, get_rgb_image


class TestLoadPreprocessImage:
    def test_rgb(self):
        img = TensorflowSlimModel._load_image(
            None, get_rgb_image())
        img = TensorflowSlimModel._preprocess_image(None, img, image_size=224)
        np.testing.assert_array_equal(img.shape, [224, 224, 3])

    def test_grayscale(self):
        img = TensorflowSlimModel._load_image(
            None, get_grayscale_image())
        img = TensorflowSlimModel._preprocess_image(None, img, image_size=224)
        np.testing.assert_array_equal(img.shape, [224, 224, 3])


class TestModels:
    def test_single_image_resnet101v2(self):
        model = TensorflowSlimModel('resnet-101_v2', batch_size=1, image_size=224)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=None)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 1

    def test_single_image_pad_resnet101v2(self):
        model = TensorflowSlimModel('resnet-101_v2', batch_size=2, image_size=224)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=None)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 1

    def test_two_batches_single_image_resnet101v2(self):
        model = TensorflowSlimModel('resnet-101_v2', batch_size=1, image_size=224)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')] * 2
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=None)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 2

    def test_two_batches_two_images_resnet101v2(self):
        model = TensorflowSlimModel('resnet-101_v2', batch_size=2, image_size=224)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')] * 4
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=None)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 4

    def test_pca_resnet101v2(self):
        model = TensorflowSlimModel('resnet-101_v2', batch_size=1, image_size=224)
        mock_imagenet(model)
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]
        layer = "resnet_v2_101/block4/unit_3/bottleneck_v2"
        activations = model.get_activations(stimuli_paths, layers=[layer], pca_components=10)
        assert isinstance(activations, NeuroidAssembly)
        assert len(activations['stimulus_path']) == 1
        assert len(activations['neuroid_id']) == 10


class TestGraph:
    def test_resnet101(self):
        model = TensorflowSlimModel('resnet-101_v2')
        graph = model.graph()
        assert 146 == len(graph.nodes)
        assert 101 == len([node_name for node_name, node in graph.nodes.items()
                         if 'conv' in node['object'].name.lower() or 'logits' in node['object'].name.lower()])
