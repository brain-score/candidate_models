import copy
import functools
import itertools
import logging
import os
from collections import OrderedDict

import h5py
import numpy as np
from multiprocessing.pool import ThreadPool
from sklearn.decomposition import PCA
from tqdm import tqdm

from brainscore.assemblies import NeuroidAssembly
from brainscore.utils import fullname
from candidate_models import s3


class Defaults(object):
    weights = 'imagenet'
    image_size = 224  # TODO: remove
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

    def __init__(self, image_size, batch_size=Defaults.batch_size):
        # require arguments here to keep the signature of different implementations the same.
        # For instance, batch_size is not required for models other than TF but by requiring it here,
        # we keep the same method signature for the caller to simplify things.
        self._image_size = image_size
        self._batch_size = batch_size
        self._logger = logging.getLogger(fullname(self))

    def get_activations(self, stimuli_paths, layers,
                        pca_components=Defaults.pca_components):
        # PCA
        get_activations = functools.partial(self._get_activations_batched, layers=layers, batch_size=self._batch_size)

        def get_image_activations_preprocessed(inputs, *args, **kwargs):
            inputs = TransformGenerator(inputs, functools.partial(self._preprocess_images, image_size=self._image_size))
            return get_activations(inputs, *args, **kwargs)

        reduce_dimensionality = self._initialize_dimensionality_reduction(pca_components,
                                                                          get_image_activations_preprocessed)
        # actual stimuli
        self._logger.info('Running stimuli')
        inputs = TransformGenerator(stimuli_paths, functools.partial(self._load_images, image_size=self._image_size))
        layer_activations = get_activations(inputs, reduce_dimensionality=reduce_dimensionality)

        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, stimuli_paths)

    def _get_activations_batched(self, inputs, layers, batch_size, reduce_dimensionality):
        layer_activations = None
        for batch_start in tqdm(range(0, len(inputs), batch_size), unit_scale=batch_size, desc="activations"):
            batch_end = min(batch_start + batch_size, len(inputs))
            self._logger.debug('Batch %d->%d/%d', batch_start, batch_end, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            batch_activations = self._get_batch_activations(batch_inputs, layer_names=layers, batch_size=batch_size)
            batch_activations = self._change_layer_activations(batch_activations, reduce_dimensionality,
                                                               keep_name=True, multithread=True)
            if layer_activations is None:
                layer_activations = copy.copy(batch_activations)
            else:
                for layer_name, layer_output in batch_activations.items():
                    layer_activations[layer_name] = np.concatenate((layer_activations[layer_name], layer_output))
        return layer_activations

    def _initialize_dimensionality_reduction(self, pca_components, get_image_activations):
        def flatten(layer_name, layer_output):
            return layer_output.reshape(layer_output.shape[0], -1)

        if pca_components is None:
            return flatten

        self._logger.info('Pre-computing principal components')
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_images = self._get_imagenet_val(pca_components)
        imagenet_activations = get_image_activations(imagenet_images, reduce_dimensionality=flatten)

        self._logger.debug('Computing ImageNet principal components')
        progress = tqdm(total=len(imagenet_activations), desc="layer principal components")

        def compute_layer_pca(activations):
            if activations.shape[1] <= pca_components:
                self._logger.debug(f"Not computing principal components for activations {activations.shape} "
                                   f"as shape is small enough already")
                pca = None
            else:
                pca = PCA(n_components=pca_components)
                pca = pca.fit(activations)
            progress.update(1)
            return pca

        pca = self._change_layer_activations(imagenet_activations, compute_layer_pca, multithread=True)
        progress.close()

        # define dimensionality reduction method for external use
        def reduce_dimensionality(layer_name, layer_activations):
            layer_activations = flatten(layer_name, layer_activations)
            if layer_activations.shape[1] < pca_components:
                self._logger.debug(f"layer {layer_name} activations are smaller than pca components: "
                                   f"{layer_activations.shape} -- not performing PCA")
                return layer_activations
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

        framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
        imagenet_filepath = os.getenv('CM_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
        if not os.path.isfile(imagenet_filepath):
            os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
            self._logger.debug(f"Downloading ImageNet validation to {imagenet_filepath}")
            s3.download_file("imagenet2012-val.hdf5", imagenet_filepath)
        with h5py.File(imagenet_filepath, 'r') as f:
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
        shapes = [a.shape for a in activations]
        self._logger.debug('Activations shapes: {}'.format(shapes))
        # layer x images x activations --> images x (layer x activations)
        activations = np.concatenate(activations, axis=-1)
        assert activations.shape[0] == len(stimuli_paths)
        assert activations.shape[1] == np.sum([np.prod(shape[1:]) for shape in shapes])
        layers = []
        for layer, shape in zip(layer_activations.keys(), shapes):
            repeated_layer = [layer] * np.prod(shape[1:])
            layers += repeated_layer
        model_assembly = NeuroidAssembly(
            activations,
            coords={'stimulus_path': stimuli_paths,
                    'neuroid_id': ('neuroid', list(range(activations.shape[1]))),
                    'layer': ('neuroid', layers)},
            dims=['stimulus_path', 'neuroid']
        )
        return model_assembly

    def _pad(self, batch_images, batch_size):
        if len(batch_images) % batch_size == 0:
            return batch_images, 0
        num_padding = batch_size - (len(batch_images) % batch_size)
        padding = np.repeat(batch_images[-1:], repeats=num_padding, axis=0)
        return np.concatenate((batch_images, padding)), num_padding

    def _unpad(self, layer_activations, num_padding):
        return self._change_layer_activations(layer_activations, lambda values: values[:-num_padding or None])

    def _change_layer_activations(self, layer_activations, change_function, keep_name=False, multithread=False):
        if not multithread:
            map_fnc = map
        else:
            pool = ThreadPool()
            map_fnc = pool.map

        def apply_change(layer_values):
            layer, values = layer_values
            values = change_function(values) if not keep_name else change_function(layer, values)
            return layer, values

        results = map_fnc(apply_change, layer_activations.items())
        results = OrderedDict(results)
        if multithread:
            pool.close()
        return results

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


class ModelLayers(dict):
    def __init__(self):
        super(ModelLayers, self).__init__()

        self['alexnet'] = \
            ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',  # conv-relu-[pool]{1,2,3,4,5}
             'classifier.2', 'classifier.5']  # fc-[relu]{6,7,8}
        self['vgg-16'] = \
            ['block{}_pool'.format(i + 1) for i in range(5)] + ['fc1', 'fc2']
        self['vgg-19'] = \
            ['block{}_pool'.format(i + 1) for i in range(5)] + ['fc1', 'fc2']
        self['squeezenet1_0'] = \
            ['features.' + layer for layer in
             ['2'] +  # max pool
             ['{}.expand3x3_activation'.format(i) for i in [3, 4, 5, 7, 8, 9, 10, 12]]  # fire outputs (ignoring pools)
             ]
        self['squeezenet1_1'] = \
            ['features.' + layer for layer in
             ['2'] +  # max pool
             ['{}.expand3x3_activation'.format(i) for i in [3, 4, 6, 7, 9, 10, 11, 12]]  # fire outputs (ignoring pools)
             ]
        self['densenet-121'] = \
            ['conv1/relu'] + \
            ['pool1'] + \
            ['conv2_block{}_concat'.format(i + 1) for i in range(6)] + \
            ['pool2_pool'] + \
            ['conv3_block{}_concat'.format(i + 1) for i in range(12)] + \
            ['pool3_pool'] + \
            ['conv4_block{}_concat'.format(i + 1) for i in range(24)] + \
            ['pool4_pool'] + \
            ['conv5_block{}_concat'.format(i + 1) for i in range(16)] + \
            ['avg_pool']
        self['densenet-169'] = \
            ['conv1/relu'] + \
            ['pool1'] + \
            ['conv2_block{}_concat'.format(i + 1) for i in range(6)] + \
            ['pool2_pool'] + \
            ['conv3_block{}_concat'.format(i + 1) for i in range(12)] + \
            ['pool3_pool'] + \
            ['conv4_block{}_concat'.format(i + 1) for i in range(32)] + \
            ['pool4_pool'] + \
            ['conv5_block{}_concat'.format(i + 1) for i in range(32)] + \
            ['avg_pool']
        self['densenet-201'] = \
            ['conv1/relu'] + \
            ['pool1'] + \
            ['conv2_block{}_concat'.format(i + 1) for i in range(6)] + \
            ['pool2_pool'] + \
            ['conv3_block{}_concat'.format(i + 1) for i in range(12)] + \
            ['pool3_pool'] + \
            ['conv4_block{}_concat'.format(i + 1) for i in range(48)] + \
            ['pool4_pool'] + \
            ['conv5_block{}_concat'.format(i + 1) for i in range(32)] + \
            ['avg_pool']
        self['xception'] = \
            ['block1_conv{}_act'.format(i + 1) for i in range(2)] + \
            ['block2_sepconv2_act'] + \
            ['block3_sepconv{}_act'.format(i + 1) for i in range(2)] + \
            ['block4_sepconv{}_act'.format(i + 1) for i in range(2)] + \
            ['block5_sepconv{}_act'.format(i + 1) for i in range(3)] + \
            ['block6_sepconv{}_act'.format(i + 1) for i in range(3)] + \
            ['block7_sepconv{}_act'.format(i + 1) for i in range(3)] + \
            ['block8_sepconv{}_act'.format(i + 1) for i in range(3)] + \
            ['block9_sepconv{}_act'.format(i + 1) for i in range(3)] + \
            ['block10_sepconv{}_act'.format(i + 1) for i in range(3)] + \
            ['block11_sepconv{}_act'.format(i + 1) for i in range(3)] + \
            ['block12_sepconv{}_act'.format(i + 1) for i in range(3)] + \
            ['block13_sepconv{}_act'.format(i + 1) for i in range(2)] + \
            ['block14_sepconv{}_act'.format(i + 1) for i in range(2)] + \
            ['avg_pool']
        self['resnet-18'] = \
            ['conv1'] + \
            ['layer1.0.relu', 'layer1.1.relu'] + \
            ['layer2.0.relu', 'layer2.0.downsample.0', 'layer2.1.relu'] + \
            ['layer3.0.relu', 'layer3.0.downsample.0', 'layer3.1.relu'] + \
            ['layer4.0.relu', 'layer4.0.downsample.0', 'layer4.1.relu'] + \
            ['avgpool']
        self['resnet-34'] = \
            ['conv1'] + \
            ['layer1.0.conv2', 'layer1.1.conv2', 'layer1.2.conv2'] + \
            ['layer2.0.downsample.0', 'layer2.1.conv2', 'layer2.2.conv2', 'layer2.3.conv2'] + \
            ['layer3.0.downsample.0', 'layer3.1.conv2', 'layer3.2.conv2', 'layer3.3.conv2',
             'layer3.4.conv2', 'layer3.5.conv2'] + \
            ['layer4.0.downsample.0', 'layer4.1.conv2', 'layer4.2.conv2'] + \
            ['avgpool']

        # Slim
        self['inception_v1'] = \
            ['MaxPool_2a_3x3'] + \
            ['Mixed_3{}'.format(i) for i in ['b', 'c']] + \
            ['Mixed_4{}'.format(i) for i in ['b', 'c', 'd', 'e', 'f']] + \
            ['Mixed_5{}'.format(i) for i in ['b', 'c']] + \
            ['AvgPool_0a_7x7']
        self['inception_v2'] = \
            ['MaxPool_2a_3x3'] + \
            ['Mixed_3{}'.format(i) for i in ['b', 'c']] + \
            ['Mixed_4{}'.format(i) for i in ['a', 'b', 'c', 'd', 'e']] + \
            ['Mixed_5{}'.format(i) for i in ['a', 'b', 'c']] + \
            ['AvgPool_1a']
        self['inception_v3'] = \
            ['Conv2d_1a_3x3', 'MaxPool_3a_3x3', 'MaxPool_5a_3x3'] + \
            ['Mixed_5{}'.format(i) for i in ['b', 'c', 'd']] + \
            ['Mixed_6{}'.format(i) for i in ['a', 'b', 'c', 'd', 'e']] + \
            ['Mixed_7{}'.format(i) for i in ['a', 'b', 'c']] + \
            ['AvgPool_1a']
        self['inception_v4'] = \
            ['Conv2d_1a_3x3'] + \
            ['Mixed_3a'] + \
            ['Mixed_4a'] + \
            ['Mixed_5{}'.format(i) for i in ['a', 'b', 'c', 'd', 'e']] + \
            ['Mixed_6{}'.format(i) for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']] + \
            ['Mixed_7{}'.format(i) for i in ['a', 'b', 'c', 'd']] + \
            ['global_pool']
        self['inception_resnet_v2'] = \
            ['Conv2d_1a_3x3', 'MaxPool_3a_3x3', 'MaxPool_5a_3x3',
             'Mixed_5b', 'Mixed_6a', 'Mixed_7a', 'Conv2d_7b_1x1', 'global_pool']
        self['resnet-50_v1'] = \
            ["resnet_v1_50/{}".format(layer) for layer in self._resnet50_layers(1)] + ['global_pool']
        self['resnet-50_v2'] = ['resnet_v2_50/block4/unit_1/bottleneck_v2']  # FIXME
            # ["resnet_v2_50/{}".format(layer) for layer in self._resnet50_layers(2)] + ['global_pool']
        self['resnet-101_v1'] = \
            ["resnet_v1_101/{}".format(layer) for layer in self._resnet101_layers(1)] + ['global_pool']
        self['resnet-101_v2'] = ['global_pool']  # FIXME\
            # ["resnet_v2_101/{}".format(layer) for layer in self._resnet101_layers(2)] + ['global_pool']
        self['resnet-152_v1'] = \
            ["resnet_v1_152/{}".format(layer) for layer in self._resnet152_layers(1)] + ['global_pool']
        self['resnet-152_v2'] = \
            ["resnet_v2_152/{}".format(layer) for layer in self._resnet152_layers(2)] + ['global_pool']
        self['nasnet_mobile'] = \
            ['Cell_{}'.format(i + 1) for i in range(-1, 3)] + ['Reduction_Cell_0'] + \
            ['Cell_{}'.format(i + 1) for i in range(3, 7)] + ['Reduction_Cell_1'] + \
            ['Cell_{}'.format(i + 1) for i in range(7, 11)] + \
            ['global_pool']
        self['nasnet_large'] = \
            ['Cell_{}'.format(i + 1) for i in range(-1, 5)] + ['Reduction_Cell_0'] + \
            ['Cell_{}'.format(i + 1) for i in range(5, 11)] + ['Reduction_Cell_1'] + \
            ['Cell_{}'.format(i + 1) for i in range(11, 17)] + \
            ['global_pool']
        self['pnasnet_large'] = \
            ['Cell_{}'.format(i + 1) for i in range(-1, 11)] + \
            ['global_pool']
        self['mobilenet_v1'] = \
            ['Conv2d_0'] + list(itertools.chain(
                *[['Conv2d_{}_depthwise'.format(i + 1), 'Conv2d_{}_pointwise'.format(i + 1)] for i in range(13)])) + \
            ['AvgPool_1a']
        self['mobilenet_v2'] = \
            ['layer_1'] + ['layer_{}/output'.format(i + 1) for i in range(1, 18)] + ['global_pool']
        self['basenet'] = \
            ['basenet-layer_v4', 'basenet-layer_pit', 'basenet-layer_ait']
        self['cornet_z'] = ['V1.output-t0', 'V2.output-t0', 'V4.output-t0', 'IT.output-t0', 'decoder.avgpool-t0']
        self['cornet_r'] = [f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                            range(5)] + ['decoder.avgpool-t0']
        self['cornet_s'] = [f'{area}.output-t{timestep}' for area, timesteps in
                            [('V4', range(4)), ('IT', range(2))] for timestep in timesteps]  # FIXME
        # ['V1.output-t0'] + \
        #                    [f'{area}.output-t{timestep}' for area, timesteps in
        #                     [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
        #                     for timestep in timesteps] + \
        #                    ['decoder.avgpool-t0']
        # self['cornet_r2'] = ['maxpool-t0'] + \  # FIXME
        #                     [f'{area}.relu3-t{timestep}' for area in ['block2', 'block3', 'block4']
        #                      for timestep in range(5)] + ['avgpool-t0']
        self['cornet_r2'] = [f'{area}.relu3-t{timestep}' for area in ['block3', 'block4']
                             for timestep in range(5)]
        """
        the last layer in each of the model's layer lists is supposed to always be the last feature layer, 
        i.e. the last layer before readout.
        """

    @staticmethod
    def _resnet50_layers(bottleneck_version):
        return ['conv1'] + \
               ["block1/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)] + \
               ["block2/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(4)] + \
               ["block3/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(6)] + \
               ["block4/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)]

    @staticmethod
    def _resnet101_layers(bottleneck_version):
        return ['conv1'] + \
               ["block1/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)] + \
               ["block2/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(4)] + \
               ["block3/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(23)] + \
               ["block4/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)]

    @staticmethod
    def _resnet152_layers(bottleneck_version):
        return ['conv1'] + \
               ["block1/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)] + \
               ["block2/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(8)] + \
               ["block3/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(36)] + \
               ["block4/unit_{}/bottleneck_v{}".format(i + 1, bottleneck_version) for i in range(3)]

    def __getitem__(self, item):
        """
        allow prefixes, e.g. mobilenet_v2 covers mobilenet_v2-123 as well as mobilenet_v2-abc
        """
        first_error = None
        for prefix in range(len(item) - 1):
            try:
                return super(ModelLayers, self).__getitem__(item[:-prefix] if prefix > 0 else item)
            except KeyError as e:
                first_error = first_error or e
        raise first_error


model_layers = ModelLayers()
