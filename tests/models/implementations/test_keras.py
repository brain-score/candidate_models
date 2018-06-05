import os

import numpy as np

from neurality.models.implementations.keras import KerasModel


class TestLoadImageKeras:
    def test_rgb(self):
        img = KerasModel._load_image(None, os.path.join(os.path.dirname(__file__), 'rgb.jpg'), image_size=224)
        np.testing.assert_array_equal(img.shape, [224, 224, 3])

    def test_grayscale(self):
        img = KerasModel._load_image(None, os.path.join(os.path.dirname(__file__), 'grayscale.png'), image_size=224)
        np.testing.assert_array_equal(img.shape, [224, 224, 3])
