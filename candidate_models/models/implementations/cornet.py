import importlib
import math
import os
import re
from collections import defaultdict

import torch
from torch import nn
from torch.nn import Module

from candidate_models.models.implementations.pytorch import PytorchModel


class TemporalPytorchModel(PytorchModel):
    def _get_activations(self, images, layer_names):
        self._layer_counter = defaultdict(lambda: 0)
        self._layer_hooks = {}
        return super(TemporalPytorchModel, self)._get_activations(images, layer_names)

    def register_hook(self, layer, layer_name, *args, **kwargs):
        # otherwise, hook is added multiple times for timesteps
        stripped_layer_name = self._strip_layer_timestep(layer_name)
        if stripped_layer_name in self._layer_hooks:
            return self._layer_hooks[stripped_layer_name]
        hook = super(TemporalPytorchModel, self).register_hook(layer, layer_name, *args, **kwargs)
        self._layer_hooks[stripped_layer_name] = hook
        return hook

    def get_layer(self, layer_name):
        layer_name = self._strip_layer_timestep(layer_name)
        return super(TemporalPytorchModel, self).get_layer(layer_name)

    def _strip_layer_timestep(self, layer_name):
        match = re.search('-t[0-9]+$', layer_name)
        if match:
            layer_name = layer_name[:match.start()]
        return layer_name

    def store_layer_output(self, layer_results, layer_name, output):
        layer_name = self._strip_layer_timestep(layer_name)
        layer_results[f"{layer_name}-t{self._layer_counter[layer_name]}"] = output.cpu().data.numpy()
        self._layer_counter[layer_name] += 1


class CORNetWrapper(TemporalPytorchModel):
    WEIGHT_MAPPING = {
        'Z': 'cornet_z_epoch25.pth.tar',
        'R': 'cornet_r_epoch25.pth.tar',
        'S': 'cornet_s_epoch43.pth.tar',
        'R2': 'cornet_r2_epoch_60.pth.tar',
    }

    def __init__(self, *args, cornet_type, **kwargs):
        self._cornet_type = cornet_type
        if cornet_type.lower() == 'r2':
            self._model_ctr = CORNetR2
        else:
            mod = importlib.import_module(f'cornet.cornet_{cornet_type.lower()}')
            self._model_ctr = getattr(mod, f'CORnet_{cornet_type.upper()}')
        super(CORNetWrapper, self).__init__(*args, **kwargs)

    def _create_model(self, weights):
        model = self._model_ctr()
        assert weights in [None, 'imagenet']
        if weights == 'imagenet':
            class Wrapper(Module):
                def __init__(self, model):
                    super(Wrapper, self).__init__()
                    self.module = model

            model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
            checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model-weights',
                                                 'cornet', self.WEIGHT_MAPPING[self._cornet_type.upper()]),
                                    map_location=lambda storage, loc: storage)  # map onto cpu
            model.load_state_dict(checkpoint['state_dict'])
            model = model.module  # unwrap
        return model


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
