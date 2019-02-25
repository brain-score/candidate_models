import logging

import functools
import importlib
import math
import os
import re
from collections import defaultdict

import torch
from torch import nn
from torch.nn import Module

from candidate_models import s3
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import LayerModel, LayerSelection, ModelCommitment

_logger = logging.getLogger(__name__)


class TemporalPytorchWrapper(PytorchWrapper):
    def get_activations(self, images, layer_names):
        # reset
        self._layer_counter = defaultdict(lambda: 0)
        self._layer_hooks = {}
        return super(TemporalPytorchWrapper, self).get_activations(images=images, layer_names=layer_names)

    def register_hook(self, layer, layer_name, target_dict):
        layer_name = self._strip_layer_timestep(layer_name)
        if layer_name in self._layer_hooks:  # add hook only once for multiple timesteps
            return self._layer_hooks[layer_name]

        def hook_function(_layer, _input, output):
            target_dict[f"{layer_name}-t{self._layer_counter[layer_name]}"] = PytorchWrapper._tensor_to_numpy(output)
            self._layer_counter[layer_name] += 1

        hook = layer.register_forward_hook(hook_function)
        self._layer_hooks[layer_name] = hook
        return hook

    def get_layer(self, layer_name):
        layer_name = self._strip_layer_timestep(layer_name)
        return super(TemporalPytorchWrapper, self).get_layer(layer_name)

    def _strip_layer_timestep(self, layer_name):
        match = re.search('-t[0-9]+$', layer_name)
        if match:
            layer_name = layer_name[:match.start()]
        return layer_name


def cornet(identifier):
    cornet_type = identifier.replace('CORnet-', '')
    if cornet_type.lower() == 'r2':
        model_ctr = CORNetR2
    else:
        mod = importlib.import_module(f'cornet.cornet_{cornet_type.lower()}')
        model_ctr = getattr(mod, f'CORnet_{cornet_type.upper()}')
    model = model_ctr()

    WEIGHT_MAPPING = {
        'Z': 'cornet_z_epoch25.pth.tar',
        'R': 'cornet_r_epoch25.pth.tar',
        'S': 'cornet_s_epoch43.pth.tar',
        'R2': 'cornet_r2_epoch_60.pth.tar',
    }

    class Wrapper(Module):
        def __init__(self, model):
            super(Wrapper, self).__init__()
            self.module = model

    model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
    framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
    weightsdir_path = os.getenv('CM_TSLIM_WEIGHTS_DIR', os.path.join(framework_home, 'model-weights', 'cornet'))
    weights_path = os.path.join(weightsdir_path, WEIGHT_MAPPING[cornet_type.upper()])
    if not os.path.isfile(weights_path):
        _logger.debug(f"Downloading weights for {identifier} to {weights_path}")
        os.makedirs(weightsdir_path, exist_ok=True)
        s3.download_file(WEIGHT_MAPPING[cornet_type.upper()], weights_path, bucket='cornet-models')
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)  # map onto cpu
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module  # unwrap

    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = TemporalPytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


class CORnetCommitment(ModelCommitment):
    def commit_region(self, region, assembly):
        layers = [layer for layer in self.layers if layer.startswith(region)]
        layer_selection = LayerSelection(model_identifier=self.layer_model.identifier,
                                         activations_model=self.layer_model.base_model, layers=layers)
        best_layer = layer_selection(assembly)
        self.layer_model.commit(region, best_layer)


class CORBlock_Rec2(nn.Module):
    scale = 6

    def __init__(self, in_channels, out_channels, ntimes=1, stride=1, h=None, name=None):
        super(CORBlock_Rec2, self).__init__()

        self.name = name
        self.ntimes = ntimes
        self.stride = stride

        self.conv_first = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.shortcut = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        # self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        # self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        for n in range(ntimes):
            setattr(self, f'bn1_{n}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'bn2_{n}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'bn3_{n}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        inp = self.conv_first(inp)
        # state = torch.zeros_like(inp)

        for n in range(self.ntimes):
            # x = inp + state # torch.stack([inp, state], dim=1)
            if n == 0:
                # x = self.conv_first(inp)
                x = inp
                residual = self.shortcut_bn(self.shortcut(inp))
                inp = residual
                # state = torch.zeros_like(inp)
            else:
                residual = inp + state
                x = residual

            # x = self.conv1(x)
            # x = getattr(self, f'bn1_{n}')(x)
            # x = self.relu1(x)

            if n == 0 and self.stride == 2:
                self.conv2.stride = (2, 2)
            else:
                self.conv2.stride = (1, 1)
            x = self.conv2(x)
            x = getattr(self, f'bn2_{n}')(x)
            x = self.relu2(x)

            x = self.conv3(x)
            x = getattr(self, f'bn3_{n}')(x)

            x += residual
            state = self.relu3(x)

        return x


class CORNetR2(nn.Module):
    def __init__(self, ntimes=(5, 5, 5), num_classes=1000):
        super(CORNetR2, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv2 = nn.Conv2d(64, , kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(128)

        self.block2 = CORBlock_Rec2(64, 128, ntimes=ntimes[0], stride=2, h=28, name='b0')
        self.block3 = CORBlock_Rec2(128, 256, ntimes=ntimes[1], stride=2, h=14, name='b1')
        self.block4 = CORBlock_Rec2(256, 512, ntimes=ntimes[2], stride=2, h=7, name='b2')

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
