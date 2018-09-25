import os
from collections import OrderedDict

import numpy as np
import pytest
from torch import nn

from candidate_models.models.implementations.pytorch import PytorchModel, PytorchPredefinedModel


class TestLoadImage:
    images = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png']

    @pytest.mark.parametrize('filename', images)
    def test_image(self, filename):
        img = PytorchModel._load_image(None, os.path.join(os.path.dirname(__file__), filename))
        assert 'RGB' == img.mode
        assert 3 == len(np.array(img).shape)
        assert 3 == np.array(img).shape[2]
        assert np.array(img).sum() > 0


class TestLayers:
    def test_alexnet(self):
        model = PytorchPredefinedModel('alexnet', weights=None, batch_size=64, image_size=224)
        layers = OrderedDict(model.layers())
        assert list(layers.keys()) == \
               [f'features.{i}' for i in range(12 + 1)] + [f'classifier.{i}' for i in range(6 + 1)]
        assert all(isinstance(layer, nn.Module) for layer in layers.values())


class TestGraph:
    def test_alexnet(self):
        model = PytorchPredefinedModel('alexnet', weights=None, batch_size=64, image_size=224)
        graph = model.graph()
        assert 20 == len(graph.nodes)
        assert 8 == len([node_name for node_name, node in graph.nodes.items()
                         if 'conv' in node['type'].__name__.lower() or 'linear' in node['type'].__name__.lower()])
