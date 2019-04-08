import glob
import logging
import os
from importlib import import_module

import functools

from brainscore.utils import LazyLoad, fullname
from candidate_models import s3
from candidate_models.base_models.cornet import cornet
from candidate_models.utils import UniqueKeyDict
from model_tools.activations import PytorchWrapper, KerasWrapper
from model_tools.activations.tensorflow import TensorflowSlimWrapper


def pytorch_model(function, image_size):
    module = import_module(f'torchvision.models')
    model_ctr = getattr(module, function)
    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=function, model=model_ctr(pretrained=True), preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper


def keras_model(module, model_function, image_size, identifier=None, model_kwargs=None):
    module = import_module(f"keras.applications.{module}")
    model_ctr, model_preprocessing = getattr(module, model_function), getattr(module, "preprocess_input")
    model = model_ctr(**(model_kwargs or {}))
    from model_tools.activations.keras import load_images
    load_preprocess = lambda image_filepaths: model_preprocessing(load_images(image_filepaths, image_size=image_size))
    wrapper = KerasWrapper(model, load_preprocess, identifier=identifier)
    wrapper.image_size = image_size
    return wrapper


class TFSlimModel:
    @staticmethod
    def init(identifier, preprocessing_type, image_size, net_name=None, labels_offset=1, batch_size=64,
             model_ctr_kwargs=None):
        import tensorflow as tf
        from nets import nets_factory

        placeholder = tf.placeholder(dtype=tf.string, shape=[batch_size])
        preprocess = TFSlimModel._init_preprocessing(placeholder, preprocessing_type, image_size=image_size)

        net_name = net_name or identifier
        model_ctr = nets_factory.get_network_fn(net_name, num_classes=labels_offset + 1000, is_training=False)
        logits, endpoints = model_ctr(preprocess, **(model_ctr_kwargs or {}))
        if 'Logits' in endpoints:  # unify capitalization
            endpoints['logits'] = endpoints['Logits']
            del endpoints['Logits']

        session = tf.Session()
        TFSlimModel._restore_imagenet_weights(identifier, session)
        wrapper = TensorflowSlimWrapper(identifier=identifier, endpoints=endpoints, inputs=placeholder, session=session,
                                        batch_size=batch_size, labels_offset=labels_offset)
        wrapper.image_size = image_size
        return wrapper

    @staticmethod
    def _init_preprocessing(placeholder, preprocessing_type, image_size):
        import tensorflow as tf
        from preprocessing import vgg_preprocessing, inception_preprocessing
        from model_tools.activations.tensorflow import load_image
        preprocessing_types = {
            'vgg': lambda image: vgg_preprocessing.preprocess_image(
                image, image_size, image_size, resize_side_min=image_size),
            'inception': lambda image: inception_preprocessing.preprocess_for_eval(
                image, image_size, image_size)
        }
        assert preprocessing_type in preprocessing_types
        preprocess_image = preprocessing_types[preprocessing_type]
        preprocess = lambda image_path: preprocess_image(load_image(image_path))
        preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)
        return preprocess

    @staticmethod
    def _restore_imagenet_weights(name, session):
        import tensorflow as tf
        var_list = None
        if name.startswith('mobilenet'):
            # Restore using exponential moving average since it produces (1.5-2%) higher accuracy according to
            # https://github.com/tensorflow/models/blob/a6494752575fad4d95e92698dbfb88eb086d8526/research/slim/nets/mobilenet/mobilenet_example.ipynb
            ema = tf.train.ExponentialMovingAverage(0.999)
            var_list = ema.variables_to_restore()
        restorer = tf.train.Saver(var_list)

        restore_path = TFSlimModel._find_model_weights(name)
        restorer.restore(session, restore_path)

    @staticmethod
    def _find_model_weights(model_name):
        _logger = logging.getLogger(fullname(TFSlimModel._find_model_weights))
        framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
        weights_path = os.getenv('CM_TSLIM_WEIGHTS_DIR', os.path.join(framework_home, 'model-weights', 'slim'))
        model_path = os.path.join(weights_path, model_name)
        if not os.path.isdir(model_path):
            _logger.debug(f"Downloading weights for {model_name} to {model_path}")
            os.makedirs(model_path)
            s3.download_folder(f"slim/{model_name}", model_path)
        fnames = glob.glob(os.path.join(model_path, '*.ckpt*'))
        assert len(fnames) > 0, f"no checkpoint found in {model_path}"
        restore_path = fnames[0].split('.ckpt')[0] + '.ckpt'
        return restore_path


def bagnet(function):
    module = import_module(f'bagnets.pytorch')
    model_ctr = getattr(module, function)
    model = model_ctr(pretrained=True)
    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=function, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def vggface():
    import keras
    weights = keras.utils.get_file(
        'rcmalli_vggface_tf_vgg16.h5',
        'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5',
        cache_subdir='models')
    wrapper = keras_model('vgg16', 'VGG16', image_size=224, identifier='vggface',
                          model_kwargs=dict(weights=weights, classes=2622))
    wrapper.image_size = 224
    return wrapper


class BaseModelPool(UniqueKeyDict):
    """
    Provides a set of standard models.
    Each entry maps from `name` to an activations extractor.
    """

    def __init__(self):
        super(BaseModelPool, self).__init__()
        self._accessed_base_models = set()

        _key_functions = {
            'alexnet': lambda: pytorch_model('alexnet', image_size=224),
            'squeezenet1_0': lambda: pytorch_model('squeezenet1_0', image_size=224),
            'squeezenet1_1': lambda: pytorch_model('squeezenet1_1', image_size=224),
            'resnet-18': lambda: pytorch_model('resnet18', image_size=224),
            'resnet-34': lambda: pytorch_model('resnet34', image_size=224),

            'vgg-16': lambda: keras_model('vgg16', 'VGG16', image_size=224),
            'vgg-19': lambda: keras_model('vgg19', 'VGG19', image_size=224),
            'vggface': vggface,
            'xception': lambda: keras_model('xception', 'Xception', image_size=299),
            'densenet-121': lambda: keras_model('densenet', 'DenseNet121', image_size=224),
            'densenet-169': lambda: keras_model('densenet', 'DenseNet169', image_size=224),
            'densenet-201': lambda: keras_model('densenet', 'DenseNet201', image_size=224),

            'inception_v1': lambda: TFSlimModel.init('inception_v1', preprocessing_type='inception', image_size=224),
            'inception_v2': lambda: TFSlimModel.init('inception_v2', preprocessing_type='inception', image_size=224),
            'inception_v3': lambda: TFSlimModel.init('inception_v3', preprocessing_type='inception', image_size=299),
            'inception_v4': lambda: TFSlimModel.init('inception_v4', preprocessing_type='inception', image_size=299),
            'inception_resnet_v2': lambda: TFSlimModel.init('inception_resnet_v2', preprocessing_type='inception',
                                                            image_size=299),
            'resnet-50_v1': lambda: TFSlimModel.init('resnet-50_v1', net_name='resnet_v1_50', preprocessing_type='vgg',
                                                     image_size=224, labels_offset=0),
            'resnet-101_v1': lambda: TFSlimModel.init('resnet-101_v1', net_name='resnet_v1_101',
                                                      preprocessing_type='vgg',
                                                      image_size=224, labels_offset=0),
            'resnet-152_v1': lambda: TFSlimModel.init('resnet-152_v1', net_name='resnet_v1_152',
                                                      preprocessing_type='vgg',
                                                      image_size=224, labels_offset=0),
            'resnet-50_v2': lambda: TFSlimModel.init('resnet-50_v2', net_name='resnet_v2_50',
                                                     preprocessing_type='inception',
                                                     image_size=299),
            'resnet-101_v2': lambda: TFSlimModel.init('resnet-101_v2', net_name='resnet_v2_101',
                                                      preprocessing_type='inception',
                                                      image_size=299),
            'resnet-152_v2': lambda: TFSlimModel.init('resnet-152_v2', net_name='resnet_v2_152',
                                                      preprocessing_type='inception',
                                                      image_size=299),
            'nasnet_mobile': lambda: TFSlimModel.init('nasnet_mobile', preprocessing_type='inception', image_size=331),
            'nasnet_large': lambda: TFSlimModel.init('nasnet_large', preprocessing_type='inception', image_size=331),
            'pnasnet_large': lambda: TFSlimModel.init('pnasnet_large', preprocessing_type='inception', image_size=331),
            'bagnet9': lambda: bagnet("bagnet9"),
            'bagnet17': lambda: bagnet("bagnet17"),
            'bagnet33': lambda: bagnet("bagnet33"),
        }
        # MobileNets
        for version, multiplier, image_size in [
            # v1
            (1, 1.0, 224), (1, 1.0, 192), (1, 1.0, 160), (1, 1.0, 128),
            (1, 0.75, 224), (1, 0.75, 192), (1, 0.75, 160), (1, 0.75, 128),
            (1, 0.5, 224), (1, 0.5, 192), (1, 0.5, 160), (1, 0.5, 128),
            (1, 0.25, 224), (1, 0.25, 192), (1, 0.25, 160), (1, 0.25, 128),
            # v2
            (2, 1.4, 224),
            (2, 1.3, 224),
            (2, 1.0, 224), (2, 1.0, 192), (2, 1.0, 160), (2, 1.0, 128), (2, 1.0, 96),
            (2, 0.75, 224), (2, 0.75, 192), (2, 0.75, 160), (2, 0.75, 128), (2, 0.75, 96),
            (2, 0.5, 224), (2, 0.5, 192), (2, 0.5, 160), (2, 0.5, 128), (2, 0.5, 96),
            (2, 0.35, 224), (2, 0.35, 192), (2, 0.35, 160), (2, 0.35, 128), (2, 0.35, 96),
        ]:
            identifier = f"mobilenet_v{version}_{multiplier}_{image_size}"
            if (version == 1 and multiplier in [.75, .5, .25]) or (version == 2 and multiplier == 1.4):
                net_name = f"mobilenet_v{version}_{multiplier * 100:03.0f}"
            else:
                net_name = f"mobilenet_v{version}"
            # arg=arg default value enforces closure:
            # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
            _key_functions[identifier] = \
                lambda identifier=identifier, image_size=image_size, net_name=net_name, \
                       multiplier=multiplier: TFSlimModel.init(
                    identifier, preprocessing_type='inception', image_size=image_size, net_name=net_name,
                    model_ctr_kwargs={'depth_multiplier': multiplier})
        # CORnets
        for cornet_type in ['Z', 'R', 'R2', 'S']:
            identifier = f"CORnet-{cornet_type}"
            _key_functions[identifier] = lambda identifier=identifier: cornet(identifier)

        # instantiate models with LazyLoad wrapper
        for identifier, function in _key_functions.items():
            self[identifier] = LazyLoad(function)

    def __getitem__(self, basemodel_identifier):
        if basemodel_identifier in self._accessed_base_models:
            raise ValueError(f"can retrieve each base model only once per session due to possible hook clashes - "
                             f"{basemodel_identifier} has already been retrieved")
        self._accessed_base_models.add(basemodel_identifier)
        return super(BaseModelPool, self).__getitem__(basemodel_identifier)


base_model_pool = BaseModelPool()
