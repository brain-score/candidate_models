import copy
import functools
import itertools
import logging
from collections import OrderedDict

import h5py
import numpy as np
from sklearn.decomposition import PCA

from mkgu.assemblies import NeuroidAssembly
from mkgu.utils import fullname


class Defaults(object):
    weights = 'imagenet'
    image_size = 224
    batch_size = 64
    pca_components = 1000


class TransformGenerator(list):
    """
    have everything operate on inputs, but transform them for retrieval
    """

    def __init__(self, inputs, transform):
        super(TransformGenerator, self).__init__(inputs)
        self._inputs = inputs
        self._transform = transform

    def __getitem__(self, item):
        return self._transform(self._inputs[item])


class DeepModel(object):
    """
    A model is defined by the model's name and image_size (defining the architecture)
    and its weights (defining the function).
    The model instantiation then relies on hyper-parameters `batch_size` and `pca_components`
    where the specific choices should not change the results too much.
    Note that `pca_components` might change the score of a value, but we are hoping
    that models with the same `pca_components` at least remain comparable.
    """

    def __init__(self, image_size=Defaults.image_size, batch_size=Defaults.batch_size):
        # require arguments here to keep the signature of different implementations the same.
        # For instance, batch_size is not required for models other than TF but by requiring it here,
        # we keep the same method signature for the caller to simplify things.
        self._image_size = image_size
        self._batch_size = batch_size
        self._logger = logging.getLogger(fullname(self))

    def get_activations(self, stimuli_paths, layers,
                        pca_components=Defaults.pca_components):
        # PCA
        get_image_activations = functools.partial(self._get_image_activations, layers=layers,
                                                  batch_size=self._batch_size, min_components=pca_components)

        def get_image_activations_preprocessed(inputs, *args, **kwargs):
            inputs = TransformGenerator(inputs, functools.partial(self._preprocess_images, image_size=self._image_size))
            return get_image_activations(inputs, *args, **kwargs)

        reduce_dimensionality = self._initialize_dimensionality_reduction(pca_components,
                                                                          get_image_activations_preprocessed)
        # actual stimuli
        inputs = TransformGenerator(stimuli_paths, functools.partial(self._load_images, image_size=self._image_size))
        layer_activations = get_image_activations(inputs, reduce_dimensionality=reduce_dimensionality)

        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, stimuli_paths)

    def _get_image_activations(self, inputs, layers, batch_size, min_components, reduce_dimensionality):
        layer_activations = self._get_activations_batched(inputs, layers, batch_size=batch_size,
                                                          reduce_dimensionality=reduce_dimensionality)
        self._pad_layers(layer_activations, min_components)
        return layer_activations

    def _get_activations_batched(self, inputs, layers, batch_size, reduce_dimensionality):
        layer_activations = None
        batch_start = 0
        while batch_start < len(inputs):
            batch_end = min(batch_start + batch_size, len(inputs))
            self._logger.debug('Batch %d->%d/%d', batch_start, batch_end, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            batch_activations = self._get_batch_activations(batch_inputs, layer_names=layers, batch_size=batch_size)
            batch_activations = self._change_layer_activations(batch_activations, reduce_dimensionality, keep_name=True)
            if layer_activations is None:
                layer_activations = copy.copy(batch_activations)
            else:
                for layer_name, layer_output in batch_activations.items():
                    layer_activations[layer_name] = np.concatenate((layer_activations[layer_name], layer_output))
            batch_start = batch_end
        return layer_activations

    def _initialize_dimensionality_reduction(self, pca_components, get_image_activations):
        def flatten(layer_name, layer_output):
            return layer_output.reshape(layer_output.shape[0], -1)

        if pca_components is None:
            return flatten

        self._logger.info('Pre-computing principal components')
        imagenet_images = self._get_imagenet_val(pca_components)
        imagenet_activations = get_image_activations(imagenet_images, reduce_dimensionality=flatten)
        pca = self._change_layer_activations(imagenet_activations,
                                             lambda activations: PCA(n_components=pca_components).fit(activations)
                                             if 0 < pca_components < np.prod(activations.shape[1:]) else activations)

        def reduce_dimensionality(layer_name, layer_activations):
            layer_activations = flatten(layer_name, layer_activations)
            return pca[layer_name].transform(layer_activations)

        return reduce_dimensionality

    def _get_imagenet_val(self, num_images):
        num_classes = 1000
        num_images_per_class = (num_images - 1) // num_classes
        base_indices = np.arange(num_images_per_class).astype(int)
        indices = []
        for i in range(num_classes):
            indices.extend(50 * i + base_indices)
        for i in range((num_images - 1) % num_classes + 1):
            indices.extend(50 * i + np.array([num_images_per_class]).astype(int))

        imagenet_file = '/braintree/data2/active/users/qbilius/datasets/imagenet2012.hdf5'
        with h5py.File(imagenet_file, 'r') as f:
            images = np.array([f['val/images'][i] for i in indices])
        return images

    def _get_batch_activations(self, images, layer_names, batch_size):
        images, num_padding = self._pad(images, batch_size)
        activations = self._get_activations(images, layer_names)
        assert isinstance(activations, OrderedDict)
        activations = self._unpad(activations, num_padding)
        return activations

    def _package(self, layer_activations, stimuli_paths):
        activations = list(layer_activations.values())
        activations = np.array(activations)
        self._logger.debug('Activations shape: {}'.format(activations.shape))
        # layer x images x activations -> images x layer x activations
        activations = activations.transpose([1, 0, 2])
        assert activations.shape[0] == len(stimuli_paths)
        assert activations.shape[1] == len(layer_activations)
        layers = np.array(list(layer_activations.keys()))
        layers = np.repeat(layers[:, np.newaxis], repeats=activations.shape[-1], axis=1)
        activations = np.reshape(activations, [activations.shape[0], np.prod(activations.shape[1:])])
        layers = np.reshape(layers, [np.prod(activations.shape[1:])])
        model_assembly = NeuroidAssembly(
            activations,
            coords={'stimulus_path': stimuli_paths,
                    'neuroid_id': ('neuroid', list(range(activations.shape[1]))),
                    'layer': ('neuroid', layers)},
            dims=['stimulus_path', 'neuroid']
        )
        return model_assembly

    def _pad_layers(self, layer_activations, num_components):
        """
        make sure all layers are the minimum size
        """
        too_small_layers = [key for key, values in layer_activations.items()
                            if num_components is not None and values[0].size < num_components]
        for layer in too_small_layers:
            self._logger.warning("Padding layer {} with zeros since its activations are too small ({})".format(
                layer, layer_activations[layer].shape))
            layer_activations[layer] = [np.pad(a, (0, num_components - a.size), 'constant', constant_values=(0,))
                                        for a in layer_activations[layer]]

    def _pad(self, batch_images, batch_size):
        if len(batch_images) % batch_size == 0:
            return batch_images, 0
        num_padding = batch_size - (batch_images.shape[0] % batch_size)
        padding = np.zeros([num_padding, *batch_images.shape[1:]]).astype(batch_images[0].dtype)
        return np.concatenate((batch_images, padding)), num_padding

    def _unpad(self, layer_activations, num_padding):
        return self._change_layer_activations(layer_activations, lambda values: values[:-num_padding or None])

    def _change_layer_activations(self, layer_activations, change_function, keep_name=False):
        return OrderedDict((layer, change_function(values) if not keep_name else change_function(layer, values))
                           for layer, values in layer_activations.items())

    def _get_activations(self, images, layer_names):
        raise NotImplementedError()

    def _load_images(self, image_filepaths, image_size):
        images = [self._load_image(image_filepath) for image_filepath in image_filepaths]
        images = self._preprocess_images(images, image_size=image_size)
        return images

    def _load_image(self, image_filepath):
        raise NotImplementedError()

    def _preprocess_images(self, images, image_size):
        raise NotImplementedError()


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
