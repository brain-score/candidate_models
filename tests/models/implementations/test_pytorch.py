import os

from candidate_models.models.implementations.pytorch import PytorchModel


class TestLoadImage:
    def test_rgb(self):
        img = PytorchModel._load_image(None, os.path.join(os.path.dirname(__file__), 'rgb.jpg'))
        assert img.mode == 'RGB'

    def test_grayscale(self):
        img = PytorchModel._load_image(None, os.path.join(os.path.dirname(__file__), 'grayscale.png'))
        assert img.mode == 'RGB'
