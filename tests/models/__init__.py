import numpy as np

from candidate_models.models.implementations import DeepModel


def mock_images(num_images):
    return np.random.rand(num_images, 224, 224, 3)


def mock_imagenet(model):
    model._get_imagenet_val = lambda num_images: mock_images(num_images)


def patch_imagenet(mocker):
    mocker.patch.object(DeepModel, '_get_imagenet_val', new=lambda self, num_images: mock_images(num_images))