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

model_layers = {
    'alexnet':
        ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
         'classifier.2', 'classifier.5', 'classifier.6'],
    'vgg16':
        ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool',
         'fc1', 'fc2'],
    'densenet':
        ['activation_1', 'activation_4', 'activation_7', 'activation_10', 'activation_13',
         'activation_16', 'activation_19', 'activation_22', 'activation_25', 'activation_28',
         'activation_31', 'activation_34', 'activation_37', 'activation_40', 'activation_43',
         'activation_46', 'activation_49', 'activation_52', 'activation_55', 'activation_58',
         'activation_61', 'activation_64', 'activation_67', 'activation_70', 'activation_73',
         'activation_76', 'activation_79', 'activation_82', 'activation_85', 'activation_88',
         'activation_91', 'activation_94', 'activation_97', 'activation_100', 'activation_103',
         'activation_106', 'activation_109', 'activation_112', 'activation_115', 'activation_118',
         'activation_121'],
    'squeezenet':
        ['pool1', 'fire2/concat', 'fire3/concat', 'fire4/concat', 'fire5/concat',
         'fire6/concat', 'fire7/concat', 'fire8/concat', 'fire9/concat', 'relu_conv10'],
    'mobilenet':
        ['conv1_relu', 'conv_dw_1_relu', 'conv_pw_1_relu',
         'conv_dw_2_relu', 'conv_pw_2_relu',
         'conv_dw_3_relu', 'conv_pw_3_relu',
         'conv_dw_4_relu', 'conv_pw_4_relu',
         'conv_dw_5_relu', 'conv_pw_5_relu',
         'conv_dw_6_relu', 'conv_pw_6_relu',
         'conv_dw_7_relu', 'conv_pw_7_relu',
         'conv_dw_8_relu', 'conv_pw_8_relu',
         'conv_dw_9_relu', 'conv_pw_9_relu',
         'conv_dw_10_relu', 'conv_pw_10_relu',
         'conv_dw_11_relu', 'conv_pw_11_relu',
         'conv_dw_12_relu', 'conv_pw_12_relu',
         'conv_dw_13_relu', 'conv_pw_13_relu',
         'global_average_pooling2d_1', 'act_softmax'],
    'resnet50':
        ['activation_1', 'activation_2', 'activation_3', 'activation_4', 'activation_5',
         'activation_6', 'activation_7', 'activation_8', 'activation_9', 'activation_10',
         'activation_11', 'activation_12', 'activation_13', 'activation_14', 'activation_15',
         'activation_16', 'activation_17', 'activation_18', 'activation_19', 'activation_20',
         'activation_21', 'activation_22', 'activation_23', 'activation_24', 'activation_25',
         'activation_26', 'activation_27', 'activation_28', 'activation_29', 'activation_30',
         'activation_31', 'activation_32', 'activation_33', 'activation_34', 'activation_35',
         'activation_36', 'activation_37', 'activation_38', 'activation_39', 'activation_40',
         'activation_41', 'activation_42', 'activation_43', 'activation_44', 'activation_45',
         'activation_46', 'activation_47', 'activation_48', 'activation_49'],
    'resnet152':
        ['relu',
         'layer1.0.relu', 'layer1.1.relu', 'layer1.2.relu',
         'layer2.0.relu', 'layer2.1.relu', 'layer2.2.relu', 'layer2.3.relu', 'layer2.4.relu',
         'layer2.5.relu', 'layer2.6.relu', 'layer2.7.relu',
         'layer3.0.relu', 'layer3.1.relu', 'layer3.2.relu', 'layer3.3.relu', 'layer3.4.relu',
         'layer3.5.relu', 'layer3.6.relu', 'layer3.7.relu', 'layer3.8.relu', 'layer3.9.relu',
         'layer3.10.relu', 'layer3.11.relu', 'layer3.12.relu', 'layer3.13.relu', 'layer3.14.relu',
         'layer3.15.relu', 'layer3.16.relu', 'layer3.17.relu', 'layer3.18.relu', 'layer3.19.relu',
         'layer3.20.relu', 'layer3.21.relu', 'layer3.22.relu', 'layer3.23.relu', 'layer3.24.relu',
         'layer3.25.relu', 'layer3.26.relu', 'layer3.27.relu', 'layer3.28.relu', 'layer3.29.relu',
         'layer3.30.relu', 'layer3.31.relu', 'layer3.32.relu', 'layer3.33.relu', 'layer3.34.relu', 'layer3.35.relu',
         'layer4.0.relu', 'layer4.1.relu', 'layer4.2.relu',
         'fc'],
    'inception_v3':
        ['mixed0', 'mixed1', 'mixed2', 'mixed3', 'mixed4', 'mixed5',
         'mixed6', 'mixed7', 'mixed8', 'mixed9', 'mixed10'],
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
        # work around to https://github.com/python-pillow/Pillow/issues/1144,
        # see https://stackoverflow.com/a/30376272/2225200
        return image.copy()


def torchvision_preprocess_input(image_size, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    from torchvision.transforms import transforms
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])
