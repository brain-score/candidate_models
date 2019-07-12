import numpy as np
from torch import nn
from typing import Dict, Tuple

from brainio_base.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from brainscore.utils import LazyLoad
from candidate_models.base_models import cornet
from result_caching import store


class CORnetCommitment:
    # TODO: get behavior back in there
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
        # cache, since piecing times together is not too fast unfortunately
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
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s222_brainmodel():
    time_start, time_step_size = 70, 100
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 2)}
    return CORnetCommitment(identifier='CORnet-S222', activations_model=cornet('CORnet-S222'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s101010_brainmodel():
    time_step_size = 20
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 10)}
    return CORnetCommitment(identifier='CORnet-S10', activations_model=cornet('CORnet-S10'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(10)), ('V4', range(10)), ('IT', range(10))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s444_brainmodel():
    time_step_size = 50
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 4)}
    return CORnetCommitment(identifier='CORnet-S444', activations_model=cornet('CORnet-S444'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(4)), ('V4', range(4)), ('IT', range(4))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s484_brainmodel():
    time_step_size = 50
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 4)}
    return CORnetCommitment(identifier='CORnet-S484', activations_model=cornet('CORnet-S484'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(4)), ('V4', range(8)), ('IT', range(4))]
                                    for timestep in timesteps] +
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
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_r_brainmodel():
    return CORnetCommitment(identifier='CORnet-R', activations_model=cornet('CORnet-R'),
                            layers=[f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                                    range(5)] + ['decoder.avgpool-t0'],
                            time_mapping={'IT': {
                                0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)}})


def cornet_r10rep_brainmodel():
    activations_model = cornet('CORnet-R')
    new_times = 10
    activations_model._model.times = new_times
    activations_model.identifier = f'CORnet-R{new_times}'
    time_step_size = 10
    time_mapping = {timestep: (70 + timestep * time_step_size, 70 + (timestep + 1) * time_step_size)
                    for timestep in range(0, new_times)}
    return CORnetCommitment(identifier=f'CORnet-R{new_times}', activations_model=activations_model,
                            layers=['maxpool-t0'] +
                                   [f'{area}.relu3-t{timestep}' for area in ['block2', 'block3', 'block4']
                                    for timestep in range(new_times)] + ['avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_r2_brainmodel():
    return CORnetCommitment(identifier='CORnet-R2', activations_model=cornet('CORnet-R2'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}' for area in ['V2', 'V4', 'IT']
                                    for timestep in range(5)] + ['avgpool-t0'],
                            time_mapping={'IT': {
                                0: (70, 105), 1: (105, 140), 2: (140, 175), 3: (175, 210), 4: (210, 250)}})


cornet_brain_pool = {
    'CORnet-S': LazyLoad(cornet_s_brainmodel),
    'CORnet-S101010': LazyLoad(cornet_s101010_brainmodel),
    'CORnet-S222': LazyLoad(cornet_s222_brainmodel),
    'CORnet-S444': LazyLoad(cornet_s444_brainmodel),
    'CORnet-S484': LazyLoad(cornet_s484_brainmodel),
    'CORnet-S10rep': LazyLoad(cornet_s10rep_brainmodel),
    'CORnet-R': LazyLoad(cornet_r_brainmodel),
    'CORnet-R10rep': LazyLoad(cornet_r10rep_brainmodel),
    'CORnet-R2': LazyLoad(cornet_r2_brainmodel),
}
