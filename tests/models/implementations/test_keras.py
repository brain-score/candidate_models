import os

import numpy as np
import pytest

from candidate_models.models.implementations.keras import KerasModel


class TestLoadImageKeras:
    images = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png']

    @pytest.mark.parametrize('filename', images)
    def test_rgb(self, filename):
        model = KerasModel('vgg-16', 'imagenet', batch_size=64, image_size=224)
        img = model._load_image(os.path.join(os.path.dirname(__file__), filename))
        img = model._preprocess_image(img, image_size=224)
        np.testing.assert_array_equal(img.shape, [224, 224, 3])


class TestGraph:
    def test_vgg16(self):
        model = KerasModel('vgg-16', 'imagenet', batch_size=64, image_size=224)
        graph = model.graph()
        assert 23 == len(graph.nodes)
        assert 16 == len([node_name for node_name, node in graph.nodes.items()
                          if 'conv' in node['type'].__name__.lower() or 'dense' in node['type'].__name__.lower()])
