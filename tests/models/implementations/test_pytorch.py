import os

import numpy as np
import pytest

from candidate_models.models.implementations.pytorch import PytorchModel


class TestLoadImage:
    images = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg']

    @pytest.mark.parametrize('filename', images)
    def test_image(self, filename):
        img = PytorchModel._load_image(None, os.path.join(os.path.dirname(__file__), filename))
        assert img.mode == 'RGB'
        assert np.asarray(img).sum() > 0


class TestGraph:
    def test_alexnet(self):
        model = PytorchModel('alexnet', weights=None, batch_size=64, image_size=224)
        graph = model.graph()
        assert 20 == len(graph.nodes)
        assert 8 == len([node_name for node_name, node in graph.nodes.items()
                         if 'conv' in node['type'].__name__.lower() or 'linear' in node['type'].__name__.lower()])
