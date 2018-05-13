import os

import numpy as np

from neurality.models.implementations import load_image_pytorch, load_image_keras


class TestLoadImagePytorch:
    def test_rgb(self):
        img = load_image_pytorch(os.path.join(os.path.dirname(__file__), 'rgb.jpg'))
        assert img.mode == 'RGB'

    def test_grayscale(self):
        img = load_image_pytorch(os.path.join(os.path.dirname(__file__), 'grayscale.png'))
        assert img.mode == 'RGB'


class TestLoadImageKeras:
    def test_rgb(self):
        img = load_image_keras(os.path.join(os.path.dirname(__file__), 'rgb.jpg'), image_size=224)
        np.testing.assert_array_equal(img.shape, [1, 224, 224, 3])

    def test_grayscale(self):
        img = load_image_keras(os.path.join(os.path.dirname(__file__), 'grayscale.png'), image_size=224)
        np.testing.assert_array_equal(img.shape, [1, 224, 224, 3])
