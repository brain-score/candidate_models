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
