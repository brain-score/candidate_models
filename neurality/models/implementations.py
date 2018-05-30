import functools
import glob
import itertools
import logging
import os
from collections import OrderedDict
from glob import iglob

import numpy as np
import pandas as pd
from PIL import Image

from neurality.models.type import get_model_type, ModelType

_logger = logging.getLogger(__name__)


def densenet121(image_size, weights='imagenet'):
    """
    https://arxiv.org/pdf/1608.06993.pdf
    """
    from keras.applications.densenet import DenseNet121, preprocess_input
    model = DenseNet121(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def densenet169(image_size, weights='imagenet'):
    """
    https://arxiv.org/pdf/1608.06993.pdf
    """
    from keras.applications.densenet import DenseNet169, preprocess_input
    model = DenseNet169(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def densenet201(image_size, weights='imagenet'):
    """
    https://arxiv.org/pdf/1608.06993.pdf
    """
    from keras.applications.densenet import DenseNet201, preprocess_input
    model = DenseNet201(input_shape=(image_size, image_size, 3), weights=weights)
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


def resnet18(image_size, weights='imagenet'):
    from torchvision.models.resnet import resnet18
    assert weights in ['imagenet', None]
    model = resnet18(pretrained=weights == 'imagenet')
    return model, torchvision_preprocess_input(image_size=image_size)


def resnet50(image_size, weights='imagenet'):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    model = ResNet50(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


MODEL_PATH = '/braintree/data2/active/users/qbilius/models/slim'
if not os.path.isdir(MODEL_PATH):
    _logger.error("\n\n\nUSING LOCAL WEIGHTS\n\n\n")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'model-weights')


class TFSlimModel(object):
    def __init__(self, model_name, batch_size=64):
        self._model_name = model_name
        self._batch_size = batch_size
        self.sess = None
        self.restore_path = None
        self.restored = False

        self.mapping = slim_models[slim_models['model'] == model_name]
        self._checkpoint = self.get_field('checkpoint')
        self._callable = self.get_field('callable')
        self._module = self.get_field('module')
        self._preprocessing = self.get_field('preprocessing')
        self._labels_offset = self.get_field('labels_offset')
        self._depth_multiplier = self.get_field('depth_multiplier')

    def get_field(self, name):
        return next(iter(self.mapping[name]))

    def __call__(self, image_size, weights='imagenet'):
        import tensorflow as tf
        from nets import nets_factory
        from preprocessing import inception_preprocessing, vgg_preprocessing

        tf.reset_default_graph()

        if self._model_name.startswith('mobilenet_v2'):
            call = 'mobilenet_v2'
            arg_scope = nets_factory.arg_scopes_map[call](weight_decay=0., is_training=False)
            kwargs = {'depth_multiplier': self._depth_multiplier}
        elif self._model_name.startswith('mobilenet_v1'):
            call = 'mobilenet_v1'
            arg_scope = nets_factory.arg_scopes_map[call](weight_decay=0., is_training=False)
            kwargs = {'depth_multiplier': self._depth_multiplier}
        else:
            call = self._callable
            arg_scope = nets_factory.arg_scopes_map[call](weight_decay=0.)
            kwargs = {}

        model = nets_factory.networks_map[call]
        inp = tf.placeholder(dtype=tf.float32,
                             shape=[self._batch_size, image_size, image_size, 3])
        if self._preprocessing == 'vgg':
            inp = tf.map_fn(lambda image: vgg_preprocessing.preprocess_image(
                tf.image.convert_image_dtype(image, dtype=tf.uint8), image_size, image_size), inp)
        else:  # inception
            inp = tf.map_fn(lambda image: inception_preprocessing.preprocess_image(
                tf.image.convert_image_dtype(image, dtype=tf.uint8), image_size, image_size), inp)

        self.inputs = inp
        with tf.contrib.slim.arg_scope(arg_scope):
            logits, endpoints = model(self.inputs,
                                      num_classes=1001 - int(self._labels_offset),
                                      is_training=False,
                                      **kwargs)
        self.endpoints = endpoints

        if self._model_name.startswith('mobilenet'):
            # Restore using exponential moving average since it produces (1.5-2%) higher accuracy
            ema = tf.train.ExponentialMovingAverage(0.999)
            var_list = ema.variables_to_restore()
        else:
            var_list = None
        self.restorer = tf.train.Saver(var_list)

        fnames = glob.glob(os.path.join(MODEL_PATH, self._model_name, '*.ckpt*'))
        assert len(fnames) > 0
        self.restore_path = fnames[0].split('.ckpt')[0] + '.ckpt'

        self.sess = self.sess or tf.Session()
        if not self.restored:
            self.restore()

        return self, lambda x: x

    def run(self, images, layer_names):
        layer_tensors = OrderedDict((layer, self.endpoints[layer]) for layer in layer_names)
        layer_outputs = self.sess.run(layer_tensors, feed_dict={self.inputs: images})
        return layer_outputs

    def restore(self):
        import tensorflow as tf
        if self._model_name.startswith('mobilenet'):
            # Restore using exponential moving average since it produces (1.5-2%) higher accuracy
            ema = tf.train.ExponentialMovingAverage(0.999)
            var_list = ema.variables_to_restore()
        else:
            var_list = None
        restorer = tf.train.Saver(var_list)
        restorer.restore(self.sess, self.restore_path)
        self.restored = True

    def __repr__(self):
        # return "\n".join("{}: {}".format(name, tensor) for name, tensor in model.endpoints.items())
        return "\n".join("{}".format(name) for name, tensor in self.endpoints.items())


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
    """
    http://arxiv.org/abs/1602.07261
    """
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
    """
    https://arxiv.org/abs/1404.5997
    """
    from torchvision.models.alexnet import alexnet
    assert weights in ['imagenet', None]
    model = alexnet(pretrained=weights == 'imagenet')
    return model, torchvision_preprocess_input(image_size=image_size)


model_mappings = {
    'alexnet': alexnet,
    'densenet-121': densenet121,
    'densenet-169': densenet169,
    'densenet-201': densenet201,
    'squeezenet': squeezenet,
    'resnet18': resnet18,
    'xception': xception,
}
slim_models = pd.read_csv(os.path.join(os.path.dirname(__file__), 'models.csv'))
slim_models = slim_models[slim_models['framework'] == 'slim']
for model in slim_models['model']:
    model_mappings[model] = TFSlimModel(model)

model_mappings['vgg-16'] = vgg16
model_mappings['vgg-19'] = vgg19

__vgg_layers = ['pool{}'.format(i + 1) for i in range(5)] + ['fc{}'.format(i + 1) for i in range(5, 7)]


def __resnet50_layers(bottleneck_version):
    return ['conv1'] + \
           ["block1/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)] + \
           ["block3/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(6)] + \
           ["block4/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)]


def __resnet101_layers(bottleneck_version):
    return ['conv1'] + \
           ["block1/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)] + \
           ["block2/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(4)] + \
           ["block3/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(23)] + \
           ["block4/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)]


def __resnet152_layers(bottleneck_version):
    return ['conv1'] + \
           ["block1/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)] + \
           ["block2/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(8)] + \
           ["block3/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(36)] + \
           ["block4/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)]


_model_layers = {
    'alexnet':
        ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',  # conv-relu-[pool]{1,2,3,4,5}
         'classifier.2', 'classifier.5'],  # fc-[relu]{6,7,8}
    'vgg-16':
        ['block{}_pool'.format(i + 1) for i in range(5)] + ['fc1', 'fc2'],
    'vgg-19':
        ['block{}_pool'.format(i + 1) for i in range(5)] + ['fc1', 'fc2'],
    'densenet-121':
        ['conv1/relu'] +
        ['pool1'] +
        ['conv2_block{}_concat'.format(i + 1) for i in range(6)] +
        ['pool2_pool'] +
        ['conv3_block{}_concat'.format(i + 1) for i in range(12)] +
        ['pool3_pool'] +
        ['conv4_block{}_concat'.format(i + 1) for i in range(24)] +
        ['pool4_pool'] +
        ['conv5_block{}_concat'.format(i + 1) for i in range(16)] +
        ['avg_pool'],
    'densenet-169':
        ['conv1/relu'] +
        ['pool1'] +
        ['conv2_block{}_concat'.format(i + 1) for i in range(6)] +
        ['pool2_pool'] +
        ['conv3_block{}_concat'.format(i + 1) for i in range(12)] +
        ['pool3_pool'] +
        ['conv4_block{}_concat'.format(i + 1) for i in range(32)] +
        ['pool4_pool'] +
        ['conv5_block{}_concat'.format(i + 1) for i in range(32)] +
        ['avg_pool'],
    'densenet-201':
        ['conv1/relu'] +
        ['pool1'] +
        ['conv2_block{}_concat'.format(i + 1) for i in range(6)] +
        ['pool2_pool'] +
        ['conv3_block{}_concat'.format(i + 1) for i in range(12)] +
        ['pool3_pool'] +
        ['conv4_block{}_concat'.format(i + 1) for i in range(48)] +
        ['pool4_pool'] +
        ['conv5_block{}_concat'.format(i + 1) for i in range(32)] +
        ['avg_pool'],
    'squeezenet':
        ['pool1'] + ['fire{}/concat'.format(i + 1) for i in range(1, 9)] +
        ['relu_conv10', 'global_average_pooling2d_1'],
    'xception':
        ['block1_conv{}_act'.format(i + 1) for i in range(2)] +
        ['block2_sepconv2_act'] +
        ['block3_sepconv{}_act'.format(i + 1) for i in range(2)] +
        ['block4_sepconv{}_act'.format(i + 1) for i in range(2)] +
        ['block5_sepconv{}_act'.format(i + 1) for i in range(3)] +
        ['block6_sepconv{}_act'.format(i + 1) for i in range(3)] +
        ['block7_sepconv{}_act'.format(i + 1) for i in range(3)] +
        ['block8_sepconv{}_act'.format(i + 1) for i in range(3)] +
        ['block9_sepconv{}_act'.format(i + 1) for i in range(3)] +
        ['block10_sepconv{}_act'.format(i + 1) for i in range(3)] +
        ['block11_sepconv{}_act'.format(i + 1) for i in range(3)] +
        ['block12_sepconv{}_act'.format(i + 1) for i in range(3)] +
        ['block13_sepconv{}_act'.format(i + 1) for i in range(2)] +
        ['block14_sepconv{}_act'.format(i + 1) for i in range(2)] +
        ['avg_pool'],
    'resnet18':
        ['conv1'] + \
        ['layer1.0.relu', 'layer1.1.relu'] + \
        ['layer2.0.relu', 'layer2.0.downsample.0', 'layer2.1.relu'] + \
        ['layer3.0.relu', 'layer3.0.downsample.0', 'layer3.1.relu'] + \
        ['layer4.0.relu', 'layer4.0.downsample.0', 'layer4.1.relu'] + \
        ['avgpool'],

    # Slim
    'inception_v1':
        ['MaxPool_2a_3x3'] +
        ['Mixed_3{}'.format(i) for i in ['b', 'c']] +
        ['Mixed_4{}'.format(i) for i in ['b', 'c', 'd', 'e', 'f']] +
        ['Mixed_5{}'.format(i) for i in ['b', 'c']] +
        ['AvgPool_0a_7x7'],
    'inception_v2':
        ['MaxPool_2a_3x3'] +
        ['Mixed_3{}'.format(i) for i in ['b', 'c']] +
        ['Mixed_4{}'.format(i) for i in ['a', 'b', 'c', 'd', 'e']] +
        ['Mixed_5{}'.format(i) for i in ['a', 'b', 'c']] +
        ['AvgPool_1a'],
    'inception_v3':
        ['Conv2d_1a_3x3', 'MaxPool_3a_3x3', 'MaxPool_5a_3x3'] +
        ['Mixed_5{}'.format(i) for i in ['b', 'c', 'd']] +
        ['Mixed_6{}'.format(i) for i in ['a', 'b', 'c', 'd', 'e']] +
        ['Mixed_7{}'.format(i) for i in ['a', 'b', 'c']] +
        ['AvgPool_1a'],
    'inception_v4':
        ['Conv2d_1a_3x3'] +
        ['Mixed_3a'] +
        ['Mixed_4a'] +
        ['Mixed_5{}'.format(i) for i in ['a', 'b', 'c', 'd', 'e']] +
        ['Mixed_6{}'.format(i) for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']] +
        ['Mixed_7{}'.format(i) for i in ['a', 'b', 'c', 'd']] +
        ['global_pool'],
    'inception_resnet_v2':
        ['Conv2d_1a_3x3', 'MaxPool_3a_3x3', 'MaxPool_5a_3x3',
         'Mixed_5b', 'Mixed_6a', 'Mixed_7a', 'Conv2d_7b_1x1', 'global_pool'],
    'resnet-50_v1':
        ["resnet_v1_50/{}".format(layer) for layer in __resnet50_layers(1)] + ['global_pool'],
    'resnet-50_v2':
        ["resnet_v2_50/{}".format(layer) for layer in __resnet50_layers(2)] + ['global_pool'],
    'resnet-101_v1':
        ["resnet_v1_101/{}".format(layer) for layer in __resnet101_layers(1)] + ['global_pool'],
    'resnet-101_v2':
        ["resnet_v2_101/{}".format(layer) for layer in __resnet101_layers(2)] + ['global_pool'],
    'resnet-152_v1':
        ["resnet_v1_152/{}".format(layer) for layer in __resnet152_layers(1)] + ['global_pool'],
    'resnet-152_v2':
        ["resnet_v2_152/{}".format(layer) for layer in __resnet152_layers(2)] + ['global_pool'],
    # 'vgg-16':
    #     ["vgg_16/{}".format(layer) for layer in __vgg_layers],
    # 'vgg-19':
    #     ["vgg_19/{}".format(layer) for layer in __vgg_layers],
    'nasnet_mobile':
        ['Cell_{}'.format(i + 1) for i in range(-1, 3)] + ['Reduction_Cell_0'] +
        ['Cell_{}'.format(i + 1) for i in range(3, 7)] + ['Reduction_Cell_1'] +
        ['Cell_{}'.format(i + 1) for i in range(7, 11)] +
        ['global_pool'],
    'nasnet_large':
        ['Cell_{}'.format(i + 1) for i in range(-1, 5)] + ['Reduction_Cell_0'] +
        ['Cell_{}'.format(i + 1) for i in range(5, 11)] + ['Reduction_Cell_1'] +
        ['Cell_{}'.format(i + 1) for i in range(11, 17)] +
        ['global_pool'],
    'pnasnet_large':
        ['Cell_{}'.format(i + 1) for i in range(-1, 11)] +
        ['global_pool'],
    ('mobilenet_v1_1.0_224', 'mobilenet_v1_1.0_192', 'mobilenet_v1_1.0_160', 'mobilenet_v1_1.0_128',
     'mobilenet_v1_0.75_224', 'mobilenet_v1_0.75_192', 'mobilenet_v1_0.75_160', 'mobilenet_v1_0.75_128',
     'mobilenet_v1_0.5_224', 'mobilenet_v1_0.5_192', 'mobilenet_v1_0.5_160', 'mobilenet_v1_0.5_128',
     'mobilenet_v1_0.25_224', 'mobilenet_v1_0.25_192', 'mobilenet_v1_0.25_160', 'mobilenet_v1_0.25_128'):
        ['Conv2d_0'] + list(itertools.chain(
            *[['Conv2d_{}_depthwise'.format(i + 1), 'Conv2d_{}_pointwise'.format(i + 1)] for i in range(13)])) +
        ['AvgPool_1a'],
    ('mobilenet_v2_1.4_224', 'mobilenet_v2_1.3_224', 'mobilenet_v2_1.0_224',
     'mobilenet_v2_1.0_192', 'mobilenet_v2_1.0_160', 'mobilenet_v2_1.0_128', 'mobilenet_v2_1.0_96',
     'mobilenet_v2_0.75_224', 'mobilenet_v2_0.75_192', 'mobilenet_v2_0.75_160', 'mobilenet_v2_0.75_128',
     'mobilenet_v2_0.75_96',
     'mobilenet_v2_0.5_224', 'mobilenet_v2_0.5_192', 'mobilenet_v2_0.5_160', 'mobilenet_v2_0.5_128',
     'mobilenet_v2_0.5_96',
     'mobilenet_v2_0.35_224', 'mobilenet_v2_0.35_192', 'mobilenet_v2_0.35_160', 'mobilenet_v2_0.35_128',
     'mobilenet_v2_0.35_96'):
        ['layer_1'] + ['layer_{}/output'.format(i + 1) for i in range(1, 18)] + ['global_pool'],
}

model_layers = {}
for model, layers in _model_layers.items():
    if isinstance(model, str):
        model_layers[model] = layers
    else:
        for _model in model:
            model_layers[_model] = layers


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
    elif model_type == ModelType.SLIM:
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
                  ModelType.PYTORCH: load_image_pytorch,
                  ModelType.SLIM: functools.partial(load_image_slim, image_size=image_size)}[model_type]
    logging.getLogger("PIL").setLevel(logging.WARNING)  # PIL in the PyTorch logs way too much
    images = [preprocess_input(load_image(image_filepath)) for image_filepath in image_filepaths]
    concat = {ModelType.KERAS: np.concatenate,
              ModelType.PYTORCH: lambda _images: Variable(torch.cat(_images)),
              ModelType.SLIM: lambda images: np.array(images)}[model_type]
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


def load_image_slim(image_filepath, image_size):
    import skimage.io
    import skimage.transform
    image = skimage.io.imread(image_filepath)
    image = skimage.transform.resize(image, (image_size, image_size))
    assert image.min() >= 0
    assert image.max() <= 1
    # image = skimage.color.gray2rgb(image)
    assert image.ndim == 3
    return image


def torchvision_preprocess_input(image_size, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    from torchvision.transforms import transforms
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])
