import os

import numpy as np

from candidate_models.models.implementations.keras import KerasModel


class TestLoadImageKeras:
    def test_rgb(self):
        model = KerasModel('vgg-16', 'imagenet', batch_size=64, image_size=224)
        img = model._load_image(os.path.join(os.path.dirname(__file__), 'rgb.jpg'))
        img = model._preprocess_image(img, image_size=224)
        np.testing.assert_array_equal(img.shape, [224, 224, 3])

    def test_grayscale(self):
        model = KerasModel('vgg-16', 'imagenet', batch_size=64, image_size=224)
        img = model._load_image(os.path.join(os.path.dirname(__file__), 'grayscale.png'))
        img = model._preprocess_image(img, image_size=224)
        np.testing.assert_array_equal(img.shape, [224, 224, 3])


class TestGraph:
    def test_vgg16(self):
        model = KerasModel('vgg-16', 'imagenet', batch_size=64, image_size=224)
        graph = model.graph()
        assert 23 == len(graph.nodes)
        assert 16 == len([node_name for node_name, node in graph.nodes.items()
                          if 'conv' in node['type'].__name__.lower() or 'dense' in node['type'].__name__.lower()])
