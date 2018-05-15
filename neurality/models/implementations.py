import functools
import logging
import os
from glob import iglob

import numpy as np

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


def vgg19(image_size, weights='imagenet'):
    from keras.applications.vgg19 import VGG19, preprocess_input
    model = VGG19(input_shape=(image_size, image_size, 3), weights=weights)
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


def resnet18(image_size, weights='imagenet'):
    from torchvision.models.resnet import resnet18
    assert weights in ['imagenet', None]
    model = resnet18(pretrained=weights == 'imagenet')
    return model, torchvision_preprocess_input(image_size=image_size)


def inception_v3(image_size, weights='imagenet'):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    model = InceptionV3(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def inception_resnet_v2(image_size, weights='imagenet'):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
    model = InceptionResNetV2(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def nasnet_large(image_size, weights='imagenet'):
    from keras.applications.nasnet import NASNetLarge, preprocess_input
    model = NASNetLarge(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def nasnet_mobile(image_size, weights='imagenet'):
    from keras.applications.nasnet import NASNetMobile, preprocess_input
    model = NASNetMobile(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def xception(image_size, weights='imagenet'):
    from keras.applications.xception import Xception, preprocess_input
    model = Xception(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def alexnet(image_size, weights='imagenet'):
    from torchvision.models.alexnet import alexnet
    assert weights in ['imagenet', None]
    model = alexnet(pretrained=weights == 'imagenet')
    return model, torchvision_preprocess_input(image_size=image_size)


model_mappings = {
    'alexnet': alexnet,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'densenet': densenet,
    'squeezenet': squeezenet,
    'mobilenet': mobilenet,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet152': resnet152,
    'inception_v3': inception_v3,
    'inception_resnet_v2': inception_resnet_v2,
    'xception': xception,
    'nasnet_large': nasnet_large,
    'nasnet_mobile': nasnet_mobile,
}

model_layers = {
    'alexnet':
        ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',  # conv-relu-[pool]{1,2,3,4,5}
         'classifier.2', 'classifier.5'],  # fc-[relu]{6,7,8}
    'vgg16':
        ['block{}_pool'.format(i + 1) for i in range(5)] + ['fc1', 'fc2'],
    'densenet':
        ['max_pooling2d_1'] + ['activation_{}'.format(i + 1) for i in range(121)] +
        ['global_average_pooling2d_1'],
    'squeezenet':
        ['pool1'] + ['fire{}/concat'.format(i + 1) for i in range(1, 9)] +
        ['relu_conv10', 'global_average_pooling2d_1'],
    'mobilenet':
        ['conv1_relu'] + list(itertools.chain(
            *[['conv_dw_{}_relu'.format(i + 1), 'conv_pw_{}_relu'.format(i + 1)] for i in range(13)])) +
        ['global_average_pooling2d_1'],
    'resnet50':
        ['activation_{}'.format(i + 1) for i in range(49)] + ['avg_pool'],
    'resnet152':
        ['relu'] +
        ['layer1.{}.relu'.format(i) for i in range(3)] +
        ['layer2.{}.relu'.format(i) for i in range(8)] +
        ['layer3.{}.relu'.format(i) for i in range(36)] +
        ['layer4.{}.relu'.format(i) for i in range(3)],
    'inception_v3':
        ['activation_{}'.format(i + 1) for i in range(10)] + ['mixed{}'.format(i) for i in range(11)],
    'inception_resnet_v2':
        ['activation_{}'.format(i + 1) for i in range(203)] + ['conv_7b_ac'],
    'nasnet_large':
        ['activation_{}'.format(i + 1) for i in range(260)],
    'nasnet_mobile':
        ['activation_{}'.format(i + 1) for i in range(188)],
}


@cache()
def create_model(model_name, model_weights, image_size):
    import torch
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
    import torch
    from torch.autograd import Variable
    load_image = {ModelType.KERAS: functools.partial(load_image_keras, image_size=image_size),
                  ModelType.PYTORCH: load_image_pytorch}[model_type]
    logging.getLogger("PIL").setLevel(logging.WARNING)  # PIL in the PyTorch logs way too much
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
    from PIL import Image
    with Image.open(image_filepath) as image:
        if image.mode.upper() != 'L':  # not binary
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return image.copy()
        # deal with binary image
        rgbimg = Image.new("RGB", image.size)
        rgbimg.paste(image)
        return rgbimg


def torchvision_preprocess_input(image_size, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    from torchvision.transforms import transforms
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])
