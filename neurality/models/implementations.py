import functools
import logging
import os
from glob import iglob

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from neurality.models.type import get_model_type, ModelType
from neurality.storage import cache

_logger = logging.getLogger(__name__)


def densenet(image_size, weights='imagenet'):
    from DenseNet import DenseNetImageNet121, preprocess_input
    model = DenseNetImageNet121(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def squeezenet(image_size, weights='imagenet'):
    from keras_squeezenet import SqueezeNet
    from keras.applications.imagenet_utils import preprocess_input
    model = SqueezeNet(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def vgg16(image_size, weights='imagenet'):
    from keras.applications.vgg16 import VGG16, preprocess_input
    model = VGG16(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def mobilenet(image_size, weights='imagenet'):
    from keras.applications.mobilenet import MobileNet, preprocess_input
    model = MobileNet(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def resnet50(image_size, weights='imagenet'):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    model = ResNet50(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def resnet152(image_size, weights='imagenet'):
    from torchvision.models.resnet import resnet152
    assert weights in ['imagenet', None]
    model = resnet152(pretrained=weights == 'imagenet')
    return model, torchvision_preprocess_input(image_size=image_size)


def inception_v3(image_size, weights='imagenet'):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    model = InceptionV3(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def inception_resnet_v2(image_size, weights='imagenet'):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
    model = InceptionResNetV2(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def nasnet(image_size, weights='imagenet'):
    from keras.applications.nasnet import NASNet, preprocess_input
    model = NASNet(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def alexnet(image_size, weights='imagenet'):
    from torchvision.models.alexnet import alexnet
    assert weights in ['imagenet', None]
    model = alexnet(pretrained=weights == 'imagenet')
    return model, torchvision_preprocess_input(image_size=image_size)


model_mappings = {
    'alexnet': alexnet,
    'vgg16': vgg16,
    'densenet': densenet,
    'squeezenet': squeezenet,
    'mobilenet': mobilenet,
    'resnet50': resnet50,
    'resnet152': resnet152,
    'inception_v3': inception_v3,
    'inception_resnet_v2': inception_resnet_v2,
    'nasnet': nasnet
}


@cache()
def create_model(model_name, model_weights, image_size):
    model, preprocessing = model_mappings[model_name](image_size, weights=model_weights)
    if get_model_type(model) == ModelType.PYTORCH and torch.cuda.is_available():
        model.cuda()
    _print_model(model, _logger.debug)
    return model, preprocessing


def _print_model(model, print_fn=print):
    model_type = get_model_type(model)
    if model_type == ModelType.KERAS:
        model.summary(print_fn=print_fn)
    elif model_type == ModelType.PYTORCH:
        print_fn(str(model))
    else:
        raise ValueError()


def prepare_images(images_directory, image_size, preprocess_input, model_type):
    image_filepaths = find_images(images_directory)
    loaded_images = load_images(image_filepaths, model_type, preprocess_input, image_size)
    return image_filepaths, loaded_images


def find_images(images_directory):
    image_filepaths = iglob(os.path.join(images_directory, '**', '*.png'), recursive=True)
    image_filepaths = list(image_filepaths)
    return image_filepaths


def load_images(image_filepaths, model_type, preprocess_input, image_size):
    load_image = {ModelType.KERAS: functools.partial(load_image_keras, image_size=image_size),
                  ModelType.PYTORCH: load_image_pytorch}[model_type]
    images = [preprocess_input(load_image(image_filepath)) for image_filepath in image_filepaths]
    concat = {ModelType.KERAS: np.concatenate,
              ModelType.PYTORCH: lambda _images: Variable(torch.cat(_images))}[model_type]
    images = concat(images)
    if model_type is ModelType.PYTORCH and torch.cuda.is_available():
        images.cuda()
    return images


def load_image_keras(image_filepath, image_size):
    from keras.preprocessing import image
    img = image.load_img(image_filepath, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def load_image_pytorch(image_filepath):
    with Image.open(image_filepath) as image:
        # work around to https://github.com/python-pillow/Pillow/issues/1144,
        # see https://stackoverflow.com/a/30376272/2225200
        return image.copy()


def torchvision_preprocess_input(image_size, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])
