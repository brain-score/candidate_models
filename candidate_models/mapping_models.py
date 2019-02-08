import itertools

from brainscore.utils import LazyLoad
from candidate_models.base_models import base_model_pool
from candidate_models.utils import UniqueKeyDict
from model_tools.activations.pca import LayerPCA
from model_tools.multilayer_mapping import LayerSelection


class ModelLayers(UniqueKeyDict):
    def __init__(self):
        super(ModelLayers, self).__init__()
        layers = {
            'alexnet':
                [  # conv-relu-[pool]{1,2,3,4,5}
                    'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
                    'classifier.2', 'classifier.5'],  # fc-[relu]{6,7,8}
            'vgg-16': [f'block{i + 1}_pool' for i in range(5)] + ['fc1', 'fc2'],
            'vgg-19': [f'block{i + 1}_pool' for i in range(5)] + ['fc1', 'fc2'],
            'squeezenet1_0':
                ['features.' + layer for layer in
                 # max pool + fire outputs (ignoring pools)
                 ['2'] + [f'{i}.expand3x3_activation' for i in [3, 4, 5, 7, 8, 9, 10, 12]]
                 ],
            'squeezenet1_1':
                ['features.' + layer for layer in
                 # max pool + fire outputs (ignoring pools)
                 ['2'] + [f'{i}.expand3x3_activation' for i in [3, 4, 6, 7, 9, 10, 11, 12]]
                 ],
            'densenet-121':
                ['conv1/relu'] + ['pool1'] +
                [f'conv2_block{i + 1}_concat' for i in range(6)] + ['pool2_pool'] +
                [f'conv3_block{i + 1}_concat' for i in range(12)] + ['pool3_pool'] +
                [f'conv4_block{i + 1}_concat' for i in range(24)] + ['pool4_pool'] +
                [f'conv5_block{i + 1}_concat' for i in range(16)] + ['avg_pool'],
            'densenet-169':
                ['conv1/relu'] + ['pool1'] +
                [f'conv2_block{i + 1}_concat' for i in range(6)] + ['pool2_pool'] +
                [f'conv3_block{i + 1}_concat' for i in range(12)] + ['pool3_pool'] +
                [f'conv4_block{i + 1}_concat' for i in range(32)] + ['pool4_pool'] +
                [f'conv5_block{i + 1}_concat' for i in range(32)] + ['avg_pool'],
            'densenet-201':
                ['conv1/relu'] + ['pool1'] +
                [f'conv2_block{i + 1}_concat' for i in range(6)] + ['pool2_pool'] +
                [f'conv3_block{i + 1}_concat' for i in range(12)] + ['pool3_pool'] +
                [f'conv4_block{i + 1}_concat' for i in range(48)] + ['pool4_pool'] +
                [f'conv5_block{i + 1}_concat' for i in range(32)] + ['avg_pool'],
            'xception':
                ['block1_conv{i + 1}_act' for i in range(2)] +
                ['block2_sepconv2_act'] +
                [f'block3_sepconv{i + 1}_act' for i in range(2)] +
                [f'block4_sepconv{i + 1}_act' for i in range(2)] +
                [f'block5_sepconv{i + 1}_act' for i in range(3)] +
                [f'block6_sepconv{i + 1}_act' for i in range(3)] +
                [f'block7_sepconv{i + 1}_act' for i in range(3)] +
                [f'block8_sepconv{i + 1}_act' for i in range(3)] +
                [f'block9_sepconv{i + 1}_act' for i in range(3)] +
                [f'block10_sepconv{i + 1}_act' for i in range(3)] +
                [f'block11_sepconv{i + 1}_act' for i in range(3)] +
                [f'block12_sepconv{i + 1}_act' for i in range(3)] +
                [f'block13_sepconv{i + 1}_act' for i in range(2)] +
                [f'block14_sepconv{i + 1}_act' for i in range(2)] +
                ['avg_pool'],
            'resnet-18':
                ['conv1'] +
                ['layer1.0.relu', 'layer1.1.relu'] +
                ['layer2.0.relu', 'layer2.0.downsample.0', 'layer2.1.relu'] +
                ['layer3.0.relu', 'layer3.0.downsample.0', 'layer3.1.relu'] +
                ['layer4.0.relu', 'layer4.0.downsample.0', 'layer4.1.relu'] +
                ['avgpool'],
            'resnet-34':
                ['conv1'] +
                ['layer1.0.conv2', 'layer1.1.conv2', 'layer1.2.conv2'] +
                ['layer2.0.downsample.0', 'layer2.1.conv2', 'layer2.2.conv2', 'layer2.3.conv2'] +
                ['layer3.0.downsample.0', 'layer3.1.conv2', 'layer3.2.conv2', 'layer3.3.conv2',
                 'layer3.4.conv2', 'layer3.5.conv2'] +
                ['layer4.0.downsample.0', 'layer4.1.conv2', 'layer4.2.conv2'] +
                ['avgpool'],

            # Slim
            'inception_v1':
                ['MaxPool_2a_3x3'] +
                [f'Mixed_3{i}' for i in ['b', 'c']] +
                [f'Mixed_4{i}' for i in ['b', 'c', 'd', 'e', 'f']] +
                [f'Mixed_5{i}' for i in ['b', 'c']] +
                ['AvgPool_0a_7x7'],
            'inception_v2':
                ['MaxPool_2a_3x3'] +
                [f'Mixed_3{i}' for i in ['b', 'c']] +
                [f'Mixed_4{i}' for i in ['a', 'b', 'c', 'd', 'e']] +
                [f'Mixed_5{i}' for i in ['a', 'b', 'c']] +
                ['AvgPool_1a'],
            'inception_v3':
                ['Conv2d_1a_3x3', 'MaxPool_3a_3x3', 'MaxPool_5a_3x3'] +
                [f'Mixed_5{i}' for i in ['b', 'c', 'd']] +
                [f'Mixed_6{i}' for i in ['a', 'b', 'c', 'd', 'e']] +
                [f'Mixed_7{i}' for i in ['a', 'b', 'c']] +
                ['AvgPool_1a'],
            'inception_v4':
                ['Conv2d_1a_3x3'] +
                ['Mixed_3a'] +
                ['Mixed_4a'] +
                [f'Mixed_5{i}' for i in ['a', 'b', 'c', 'd', 'e']] +
                [f'Mixed_6{i}' for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']] +
                [f'Mixed_7{i}' for i in ['a', 'b', 'c', 'd']] +
                ['global_pool'],
            'inception_resnet_v2':
                ['Conv2d_1a_3x3', 'MaxPool_3a_3x3', 'MaxPool_5a_3x3',
                 'Mixed_5b', 'Mixed_6a', 'Mixed_7a', 'Conv2d_7b_1x1', 'global_pool'],
            'resnet-50_v1': [f"resnet_v1_50/{layer}" for layer in self._resnet50_layers(1)] + ['global_pool'],
            'resnet-50_v2': [f"resnet_v2_50/{layer}" for layer in self._resnet50_layers(2)] + ['global_pool'],
            'resnet-101_v1': [f"resnet_v1_101/{layer}" for layer in self._resnet101_layers(1)] + ['global_pool'],
            'resnet-101_v2': [f"resnet_v2_101/{layer}" for layer in self._resnet101_layers(2)] + ['global_pool'],
            'resnet-152_v1': [f"resnet_v1_152/{layer}" for layer in self._resnet152_layers(1)] + ['global_pool'],
            'resnet-152_v2': [f"resnet_v2_152/{layer}" for layer in self._resnet152_layers(2)] + ['global_pool'],
            'nasnet_mobile':
                [f'Cell_{i + 1}' for i in range(-1, 3)] + ['Reduction_Cell_0'] +
                [f'Cell_{i + 1}' for i in range(3, 7)] + ['Reduction_Cell_1'] +
                [f'Cell_{i + 1}' for i in range(7, 11)] + ['global_pool'],
            'nasnet_large':
                [f'Cell_{i + 1}' for i in range(-1, 5)] + ['Reduction_Cell_0'] +
                [f'Cell_{i + 1}' for i in range(5, 11)] + ['Reduction_Cell_1'] +
                [f'Cell_{i + 1}' for i in range(11, 17)] + ['global_pool'],
            'pnasnet_large':
                [f'Cell_{i + 1}' for i in range(-1, 11)] + ['global_pool'],
            'mobilenet_v1':
                ['Conv2d_0'] + list(itertools.chain(
                    *[[f'Conv2d_{i + 1}_depthwise', f'Conv2d_{i + 1}_pointwise'] for i in range(13)])) +
                ['AvgPool_1a'],
            'mobilenet_v2': ['layer_1'] + [f'layer_{i + 1}/output' for i in range(1, 18)] + ['global_pool'],
            'basenet': ['basenet-layer_v4', 'basenet-layer_pit', 'basenet-layer_ait'],
        }
        for basemodel_key, default_layers in layers.items():
            self[basemodel_key] = default_layers

    @staticmethod
    def _resnet50_layers(bottleneck_version):
        return ModelLayers._resnet_layers(bottleneck_version=bottleneck_version, units=[3, 4, 6, 3])

    @staticmethod
    def _resnet101_layers(bottleneck_version):
        return ModelLayers._resnet_layers(bottleneck_version=bottleneck_version, units=[3, 4, 23, 3])

    @staticmethod
    def _resnet152_layers(bottleneck_version):
        return ModelLayers._resnet_layers(bottleneck_version=bottleneck_version, units=[3, 8, 36, 3])

    @staticmethod
    def _resnet_layers(bottleneck_version, units):
        return ['conv1'] + \
               [f"block{block + 1}/unit_{unit + 1}/bottleneck_v{bottleneck_version}"
                for block, block_units in enumerate(units) for unit in range(block_units)]


model_layers = ModelLayers()


class ModelLayersPool(UniqueKeyDict):
    def __init__(self):
        super(ModelLayersPool, self).__init__()
        for basemodel_key, layers in model_layers.items():
            # enforce early parameter binding: https://stackoverflow.com/a/3431699/2225200
            def load(basemodel_key=basemodel_key):
                activations_model = base_model_pool[basemodel_key]
                pca_components = 1000
                LayerPCA.hook(activations_model, n_components=pca_components)
                return activations_model

            self[basemodel_key] = {'model': LazyLoad(load), 'layers': layers}


model_layers_pool = ModelLayersPool()


class MappingModelPool(UniqueKeyDict):
    def __init__(self):
        super(MappingModelPool, self).__init__()
        for basemodel_key, default_layers in model_layers.items():
            # enforce early parameter binding: https://stackoverflow.com/a/3431699/2225200
            def load(basemodel_key=basemodel_key, default_layers=default_layers):
                activations_model = base_model_pool[basemodel_key]
                pca_components = 1000
                LayerPCA.hook(activations_model, n_components=pca_components)
                return LayerSelection(model_identifier=f"{activations_model.identifier}-pca_{pca_components}",
                                      activations_model=activations_model, layers=default_layers)

            self[basemodel_key] = LazyLoad(load)


mapping_model_pool = MappingModelPool()
