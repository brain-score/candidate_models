from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

from candidate_models.models.implementations import DeepModel


def mock_image():
    return np.random.rand(224, 224, 3)


def mock_imagenet(model):
    model._get_imagenet_val = _get_imagenet_val


def _get_imagenet_val(num_images):
    image = mock_image()
    file = NamedTemporaryFile(suffix='.png', delete=False)
    Image.fromarray((255 * image).astype(np.uint8)).save(file.name)
    return [file.name] * num_images



def patch_imagenet(mocker):
    mocker.patch.object(DeepModel, '_get_imagenet_val', new=lambda self, num_images: _get_imagenet_val(num_images))
