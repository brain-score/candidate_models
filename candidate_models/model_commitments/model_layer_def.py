import itertools

from brainscore.submission.ml_pool import ModelLayers

def resnet_pt_layers(units):
    return (['relu', 'maxpool'] +
            [f"layer1.{i}" for i in range(units[0])] +
            [f"layer2.{i}" for i in range(units[1])] +
            [f"layer3.{i}" for i in range(units[2])] +
            [f"layer4.{i}" for i in range(units[3])] +
            ["avgpool"])

def resnet50_layers(bottleneck_version):
    return resnet_layers(bottleneck_version=bottleneck_version, units=[3, 4, 6, 3])


def resnet101_layers(bottleneck_version):
    return resnet_layers(bottleneck_version=bottleneck_version, units=[3, 4, 23, 3])


def resnet152_layers(bottleneck_version):
    return resnet_layers(bottleneck_version=bottleneck_version, units=[3, 8, 36, 3])


def resnet_layers(bottleneck_version, units):
    return ['conv1'] + \
           [f"block{block + 1}/unit_{unit + 1}/bottleneck_v{bottleneck_version}"
            for block, block_units in enumerate(units) for unit in range(block_units)]


def resnext101_layers():
    return (['conv1'] +
            # note that while relu is used multiple times, by default the last one will overwrite all previous ones
            [f"layer{block + 1}.{unit}.relu"
             for block, block_units in enumerate([3, 4, 23, 3]) for unit in range(block_units)] +
            ['avgpool'])


def mobilenet_v1():
    return ['Conv2d_0'] + list(itertools.chain(
        *[[f'Conv2d_{i + 1}_depthwise', f'Conv2d_{i + 1}_pointwise'] for i in range(13)])) + ['AvgPool_1a']


def mobilenet_v2():
    return ['layer_1'] + [f'layer_{i + 1}/output' for i in range(1, 18)] + ['global_pool']


def bagnet():
    return (['relu'] +
            [f'layer{layer + 1}.{block}.relu' for layer, blocks in
             enumerate([2, 3, 5, 2]) for block in range(blocks + 1)] +
            ['avgpool'])


def unsup_vvs_res18():
    return ['encode_1.conv'] + [f'encode_{i}' for i in range(1, 10)]


def unsup_vvs_pt_res18():
    return ['relu', 'maxpool'] +\
           ['layer1.0.relu', 'layer1.1.relu'] +\
           ['layer2.0.relu', 'layer2.1.relu'] +\
           ['layer3.0.relu', 'layer3.1.relu'] +\
           ['layer4.0.relu', 'layer4.1.relu']


def prednet():
    num_layers = 4
    return ['A_%i' % i for i in range(1, num_layers)] \
            + ['Ahat_%i' % i for i in range(1, num_layers)] \
            + ['E_%i' % i for i in range(1, num_layers)] \
            + ['R_%i' % i for i in range(1, num_layers)]


layers = {
    'alexnet':
        [  # conv-relu-[pool]{1,2,3,4,5}
            'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
            'classifier.2', 'classifier.5'],  # fc-[relu]{6,7,8}
    'vgg-16': [f'block{i + 1}_pool' for i in range(5)] + ['fc1', 'fc2'],
    'vgg-19': [f'block{i + 1}_pool' for i in range(5)] + ['fc1', 'fc2'],
    'vgg-11-pt': [f'features.{i}' for i in [1,2,4,5,7,9,10,12,14,15,17,19,20]] + ['classifier.1', 'classifier.4'],
    'vgg-11-bn-pt': [f'features.{i}' for i in [2,3,6,7,10,13,14,17,20,21,24,27,28]] + ['classifier.1', 'classifier.4'],
    'vgg-13-pt': [f'features.{i}' for i in [1,3,4,6,8,9,11,13,14,16,18,19,21,23,24]] + ['classifier.1', 'classifier.4'],
    'vgg-13-bn-pt': [f'features.{i}' for i in [2,5,6,9,12,13,16,19,20,23,26,27,30,33,34]] + ['classifier.1', 'classifier.4'],
    'vgg-16-pt': [f'features.{i}' for i in [1,3,4,6,8,9,11,13,15,16,18,20,21,23,25,27,29,30]] + ['classifier.1', 'classifier.4'],
    'vgg-16-bn-pt': [f'features.{i}' for i in [2,5,6,9,12,13,16,19,22,23,26,29,32,33,36,39,42,43]] + ['classifier.1', 'classifier.4'],
    'vgg-19-pt': [f'features.{i}' for i in [1,3,4,6,8,9,11,13,15,17,18,20,22,24,26,27,29,31,33,35,36]] + ['classifier.1', 'classifier.4'],
    'vgg-19-bn-pt': [f'features.{i}' for i in [2,5,6,9,12,13,16,19,22,25,26,29,32,35,38,39,42,45,48,51,52]] + ['classifier.1', 'classifier.4'],
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
        [f'block1_conv{i + 1}_act' for i in range(2)] +
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
    'resnet-18-pt': resnet_pt_layers([2,2,2,2]),
    'resnet-34-pt': resnet_pt_layers([3,4,6,3]),
    'resnet-50-pt': resnet_pt_layers([3,4,6,3]),
    'wide-resnet-50-pt': resnet_pt_layers([3,4,6,3]),
    'resnet-101-pt': resnet_pt_layers([3,4,23,3]),
    'wide-resnet-101-pt': resnet_pt_layers([3,4,23,3]),
    'resnet-152-pt': resnet_pt_layers([3,8,36,3]),
    'resnext-50-32x4d-pt': resnet_pt_layers([3,4,6,3]),
    'resnext-101-32x8d-pt': resnet_pt_layers([3,4,23,3]),
    'resnet-50-robust-l2-3': resnet_pt_layers([3,4,6,3]),
    'resnet-50-robust-linf-4': resnet_pt_layers([3,4,6,3]),
    'resnet-50-robust-linf-8': resnet_pt_layers([3,4,6,3]),
    'resnet-50-GNTsig0.5': resnet_pt_layers([3,4,6,3]),
    'resnet-50-ANT3x3_SIN': resnet_pt_layers([3,4,6,3]),
    'resnet-50-SIN': resnet_pt_layers([3,4,6,3]),
    'resnet-50-SIN_IN': resnet_pt_layers([3,4,6,3]),
    'resnet-50-SIN_IN_IN': resnet_pt_layers([3,4,6,3]),
    'voneresnet-50':
        ['vone_block'] +
        ['model.layer1.0', 'model.layer1.1', 'model.layer1.2'] +
        ['model.layer2.0', 'model.layer2.1', 'model.layer2.2', 'model.layer2.3'] +
        ['model.layer3.0', 'model.layer3.1', 'model.layer3.2', 'model.layer3.3',
         'model.layer3.4', 'model.layer3.5'] +
        ['model.layer4.0', 'model.layer4.1', 'model.layer4.2'] +
        ['model.avgpool'],
    'voneresnet-50-robust':
        ['vone_block'] +
        ['model.layer1.0', 'model.layer1.1', 'model.layer1.2'] +
        ['model.layer2.0', 'model.layer2.1', 'model.layer2.2', 'model.layer2.3'] +
        ['model.layer3.0', 'model.layer3.1', 'model.layer3.2', 'model.layer3.3',
         'model.layer3.4', 'model.layer3.5'] +
        ['model.layer4.0', 'model.layer4.1', 'model.layer4.2'] +
        ['model.avgpool'],
    'densenet-121-pt':
        ['features.relu0', 'features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.relu1' for i in range(6)] + 
        ['features.transition1.relu'] +
        [f'features.denseblock2.denselayer{i + 1}.relu1' for i in range(12)] + 
        ['features.transition2.relu'] +
        [f'features.denseblock3.denselayer{i + 1}.relu1' for i in range(24)] + 
        ['features.transition3.relu'] +
        [f'features.denseblock4.denselayer{i + 1}.relu1' for i in range(16)],
    'densenet-169-pt':
        ['features.relu0', 'features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.relu1' for i in range(6)] + 
        ['features.transition1.relu'] +
        [f'features.denseblock2.denselayer{i + 1}.relu1' for i in range(12)] + 
        ['features.transition2.relu'] +
        [f'features.denseblock3.denselayer{i + 1}.relu1' for i in range(32)] + 
        ['features.transition3.relu'] +
        [f'features.denseblock4.denselayer{i + 1}.relu1' for i in range(32)],
    'densenet-201-pt':
        ['features.relu0', 'features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.relu1' for i in range(6)] + 
        ['features.transition1.relu'] +
        [f'features.denseblock2.denselayer{i + 1}.relu1' for i in range(12)] + 
        ['features.transition2.relu'] +
        [f'features.denseblock3.denselayer{i + 1}.relu1' for i in range(48)] + 
        ['features.transition3.relu'] +
        [f'features.denseblock4.denselayer{i + 1}.relu1' for i in range(32)],
    'densenet-161-pt':
        ['features.relu0', 'features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.relu2' for i in range(6)] + 
        ['features.transition1.relu'] +
        [f'features.denseblock2.denselayer{i + 1}.relu2' for i in range(12)] + 
        ['features.transition2.relu'] +
        [f'features.denseblock3.denselayer{i + 1}.relu2' for i in range(36)] + 
        ['features.transition3.relu'] +
        [f'features.denseblock4.denselayer{i + 1}.relu2' for i in range(24)],
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
    'resnet-50_v1': [f"resnet_v1_50/{layer}" for layer in resnet50_layers(1)] + ['global_pool'],
    'resnet-50_v2': [f"resnet_v2_50/{layer}" for layer in resnet50_layers(2)] + ['global_pool'],
    'resnet-101_v1': [f"resnet_v1_101/{layer}" for layer in resnet101_layers(1)] + ['global_pool'],
    'resnet-101_v2': [f"resnet_v2_101/{layer}" for layer in resnet101_layers(2)] + ['global_pool'],
    'resnet-152_v1': [f"resnet_v1_152/{layer}" for layer in resnet152_layers(1)] + ['global_pool'],
    'resnet-152_v2': [f"resnet_v2_152/{layer}" for layer in resnet152_layers(2)] + ['global_pool'],
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
    'bagnet9': bagnet(),
    'bagnet17': bagnet(),
    'bagnet33': bagnet(),
    'resnext101_32x8d_wsl': resnext101_layers(),
    'resnext101_32x16d_wsl': resnext101_layers(),
    'resnext101_32x32d_wsl': resnext101_layers(),
    'resnext101_32x48d_wsl': resnext101_layers(),
    'fixres_resnext101_32x48d_wsl': resnext101_layers(),
    'dcgan': ['main.0', 'main.2', 'main.5', 'main.8', 'main.12'],
    # ConvRNNs
    'convrnn_224': ['logits'],
    # Unsupervised VVS
    'resnet18-supervised': unsup_vvs_res18(),
    'resnet18-local_aggregation': unsup_vvs_res18(),
    'resnet18-instance_recognition': unsup_vvs_res18(),
    'resnet18-autoencoder': unsup_vvs_res18(),
    'resnet18-contrastive_predictive': unsup_vvs_res18(),
    'resnet18-colorization': unsup_vvs_res18(),
    'resnet18-relative_position': unsup_vvs_res18(),
    'resnet18-depth_prediction': unsup_vvs_res18(),
    'prednet': prednet(),
    'resnet18-simclr': unsup_vvs_res18()[1:],
    'resnet18-deepcluster': unsup_vvs_pt_res18(),
    'resnet18-contrastive_multiview': unsup_vvs_pt_res18(),
}

model_layers = ModelLayers(layers)
model_layers['vggface'] = model_layers['vgg-16']

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
    if version == 1:
        model_layers[identifier] = mobilenet_v1()
    else:
        model_layers[identifier] = mobilenet_v2()
