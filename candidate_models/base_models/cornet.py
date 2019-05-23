import importlib
import logging
import os
from collections import defaultdict

import functools
import math
import numpy as np
import re
import torch
from torch import nn
from torch.nn import Module
from typing import Dict, Tuple

from brainio_base.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from candidate_models import s3
from model_tools.activations.pytorch import PytorchWrapper
from result_caching import store

_logger = logging.getLogger(__name__)


class TemporalPytorchWrapper(PytorchWrapper):
    def __call__(self, *args, **kwargs):
        activations = super(TemporalPytorchWrapper, self).__call__(*args, **kwargs)
        # introduce time dimension
        regions = defaultdict(list)
        for layer in set(activations['layer'].values):
            match = re.match(r'([^-]*)\.output-t([0-9]+)', layer)
            region, timestep = match.group(1), match.group(2)
            regions[region].append((layer, timestep))
        activations = {(region, timestep): activations.sel(layer=time_layer)
                       for region, time_layers in regions.items() for (time_layer, timestep) in time_layers}
        for key in activations:
            region, timestep = key
            exclude_coords = ['neuroid_id']  # otherwise, neuroid dim will be as large as before with nans
            activations[key]['region'] = 'neuroid', [region] * len(activations[key]['neuroid'])
            activations[key] = NeuroidAssembly([activations[key].values], coords={
                **{coord: (dims, values) for coord, dims, values in walk_coords(activations[key])
                   if coord not in exclude_coords},
                **{'time_step': [int(timestep)]}
            }, dims=['time_step'] + list(activations[key].dims))
        activations = list(activations.values())
        activations = merge_data_arrays(activations)
        return activations

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
    cornet_type = re.match('CORnet-(.*)', identifier).group(1)
    if cornet_type.lower() == 'r2':
        model_ctr = CORNetR2
    elif cornet_type.lower() == 's10':
        from .cornet_s_10 import CORnet_S as CORnet_S10
        model_ctr = CORnet_S10
    elif cornet_type.lower() == 's222':
        from .cornet_s_222 import CORnet_S as CORnet_S222
        model_ctr = CORnet_S222
    elif cornet_type.lower() == 's444':
        from .cornet_s_444 import CORnet_S as CORnet_S444
        model_ctr = CORnet_S444
    elif cornet_type.lower() == 's484':
        from .cornet_s_484 import CORnet_S as CORnet_S484
        model_ctr = CORnet_S484
    else:
        mod = importlib.import_module(f'cornet.cornet_{cornet_type.lower()}')
        model_ctr = getattr(mod, f'CORnet_{cornet_type.upper()}')
    model = model_ctr()

    WEIGHT_MAPPING = {
        'Z': 'cornet_z_epoch25.pth.tar',
        'R': 'cornet_r_epoch25.pth.tar',
        'S': 'cornet_s_epoch43.pth.tar',
        'R2': 'cornet_r2_epoch_60.pth.tar',
        'S10': 'cornet_s10_latest.pth.tar',
        'S222': 'cornet_s222_latest.pth.tar',
        'S444': 'cornet_s444_latest.pth.tar',
        'S484': 'cornet_s484_latest.pth.tar',
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


class CORnetCommitment:
    def __init__(self, identifier, activations_model, layers,
                 time_mapping: Dict[str, Dict[int, Tuple[int, int]]], behavioral_readout_layer=None):
        """
        :param time_mapping: mapping from region -> {model_timestep -> (time_bin_start, time_bin_end)}
        """
        self.layers = layers
        self.region_assemblies = {}
        self.activations_model = activations_model
        self.time_mapping = time_mapping
        self.do_behavior = False
        self.recording_layers = None
        self.recording_time_bins = None

    def commit_region(self, region, assembly):
        pass  # already anatomically pre-mapped

    def start_recording(self, recording_target, time_bins):
        self.recording_layers = [layer for layer in self.layers if layer.startswith(recording_target)]
        self.recording_time_bins = time_bins

    def look_at(self, stimuli):
        return self.look_at_cached(self.activations_model.identifier, stimuli.name, stimuli)

    @store(identifier_ignore=['stimuli'])
    def look_at_cached(self, activations_model_identifier, stimuli_identifier, stimuli):
        responses = self.activations_model(stimuli, layers=self.recording_layers)
        # map time
        regions = set(responses['region'].values)
        if len(regions) > 1:
            raise NotImplementedError("cannot handle more than one simultaneous region")
        region = list(regions)[0]
        time_bins = [self.time_mapping[region][timestep] for timestep in responses['time_step'].values]
        responses['time_bin_start'] = 'time_step', [time_bin[0] for time_bin in time_bins]
        responses['time_bin_end'] = 'time_step', [time_bin[1] for time_bin in time_bins]
        responses = NeuroidAssembly(responses.rename({'time_step': 'time_bin'}))
        # select time
        time_responses = []
        for time_bin in self.recording_time_bins:
            time_bin = time_bin if not isinstance(time_bin, np.ndarray) else time_bin.tolist()
            time_bin_start, time_bin_end = time_bin
            nearest_start = find_nearest(responses['time_bin_start'].values, time_bin_start)
            bin_responses = responses.sel(time_bin_start=nearest_start)
            bin_responses = NeuroidAssembly(bin_responses.values, coords={
                **{coord: (dims, values) for coord, dims, values in walk_coords(bin_responses)
                   if coord not in ['time_bin_level_0', 'time_bin_end']},
                **{'time_bin_start': ('time_bin', [time_bin_start]),
                   'time_bin_end': ('time_bin', [time_bin_end])}
            }, dims=bin_responses.dims)
            time_responses.append(bin_responses)
        responses = merge_data_arrays(time_responses)
        return responses


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def cornet_s_brainmodel():
    # time_start, time_step_size = 70, 100
    time_start, time_step_size = 100, 100
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 2)}
    return CORnetCommitment(identifier='CORnet-S', activations_model=cornet('CORnet-S'),
                            layers=['V1.output-t0'] + \
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] + \
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s222_brainmodel():
    time_start, time_step_size = 70, 100
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 2)}
    return CORnetCommitment(identifier='CORnet-S222', activations_model=cornet('CORnet-S222'),
                            layers=['V1.output-t0'] + \
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] + \
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s10_brainmodel():
    time_step_size = 20
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 10)}
    return CORnetCommitment(identifier='CORnet-S10', activations_model=cornet('CORnet-S10'),
                            layers=['V1.output-t0'] + \
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(10)), ('V4', range(10)), ('IT', range(10))]
                                    for timestep in timesteps] + \
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s444_brainmodel():
    time_step_size = 50
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 4)}
    return CORnetCommitment(identifier='CORnet-S444', activations_model=cornet('CORnet-S444'),
                            layers=['V1.output-t0'] + \
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(4)), ('V4', range(4)), ('IT', range(4))]
                                    for timestep in timesteps] + \
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s484_brainmodel():
    time_step_size = 50
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 4)}
    return CORnetCommitment(identifier='CORnet-S484', activations_model=cornet('CORnet-S484'),
                            layers=['V1.output-t0'] + \
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(4)), ('V4', range(8)), ('IT', range(4))]
                                    for timestep in timesteps] + \
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s10rep_brainmodel():
    activations_model = cornet('CORnet-S')
    old_times = activations_model._model.IT.times
    new_times = 10
    activations_model._model.IT.times = new_times
    size_12 = activations_model._model.IT.norm1_0.num_features
    size_3 = activations_model._model.IT.norm3_0.num_features
    for t in range(old_times, new_times):
        setattr(activations_model._model.IT, f'norm1_{t}', nn.BatchNorm2d(size_12))
        setattr(activations_model._model.IT, f'norm2_{t}', nn.BatchNorm2d(size_12))
        setattr(activations_model._model.IT, f'norm3_{t}', nn.BatchNorm2d(size_3))
    identifier = f'CORnet-S{new_times}rep'
    activations_model.identifier = identifier
    time_step_size = 10
    time_mapping = {timestep: (70 + timestep * time_step_size, 70 + (timestep + 1) * time_step_size)
                    for timestep in range(0, new_times)}
    return CORnetCommitment(identifier=identifier, activations_model=activations_model,
                            layers=['V1.output-t0'] + \
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] + \
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_r_brainmodel():
    return CORnetCommitment(identifier='CORnet-R', activations_model=cornet('CORnet-R'),
                            layers=[f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                                    range(5)] + ['decoder.avgpool-t0'],
                            time_mapping={'IT': {
                                0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)}})


def cornet_r10_brainmodel():
    activations_model = cornet('CORnet-R')
    new_times = 10
    activations_model._model.times = new_times
    activations_model.identifier = f'CORnet-R{new_times}'
    time_step_size = 10
    time_mapping = {timestep: (70 + timestep * time_step_size, 70 + (timestep + 1) * time_step_size)
                    for timestep in range(0, new_times)}
    return CORnetCommitment(identifier=f'CORnet-R{new_times}', activations_model=activations_model,
                            layers=['maxpool-t0'] + \
                                   [f'{area}.relu3-t{timestep}' for area in ['block2', 'block3', 'block4']
                                    for timestep in range(new_times)] + ['avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_r2_brainmodel():
    # TODO: layers don't start with region
    return CORnetCommitment(identifier='CORnet-R2', activations_model=cornet('CORnet-R2'),
                            layers=['maxpool-t0'] + \
                                   [f'{area}.relu3-t{timestep}' for area in ['block2', 'block3', 'block4']
                                    for timestep in range(5)] + ['avgpool-t0'],
                            time_mapping={'IT': {
                                0: (70, 105), 1: (105, 140), 2: (140, 175), 3: (175, 210), 4: (210, 250)}})


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
