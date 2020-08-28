
from candidate_models.model_commitments.cornets import CORnetCommitment, _build_time_mappings, find_nearest
import logging

import numpy as np
from tqdm import tqdm

from brainio_base.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from candidate_models.base_models import vonecornet
from result_caching import store
from brainscore.submission.utils import UniqueKeyDict
from brainscore.utils import LazyLoad
_logger = logging.getLogger(__name__)


class VOneCORnetCommitment(CORnetCommitment):
    def start_recording(self, recording_target, time_bins):
        if recording_target == 'V1':
            self.recording_layers = ['vone_block.output-t0']
        else:
            self.recording_layers = [layer for layer in self.layers if recording_target in layer]
        self.recording_time_bins = time_bins

    @store(identifier_ignore=['stimuli'])
    def look_at_cached(self, model_identifier, stimuli_identifier, stimuli):
        responses = self.activations_model(stimuli, layers=self.recording_layers)
        # map time
        regions = set(responses['region'].values)
        if len(regions) > 1:
            raise NotImplementedError("cannot handle more than one simultaneous region")
        region = list(regions)[0]

        if '.' in region:
            region = region.split('.')[1]
        if region == 'vone_block':
            region = 'V1'

        time_bins = [self.time_mapping[region][timestep] if timestep in self.time_mapping[region] else (None, None)
                     for timestep in responses['time_step'].values]
        responses['time_bin_start'] = 'time_step', [time_bin[0] for time_bin in time_bins]
        responses['time_bin_end'] = 'time_step', [time_bin[1] for time_bin in time_bins]
        responses = NeuroidAssembly(responses.rename({'time_step': 'time_bin'}))
        responses = responses[{'time_bin': [not np.isnan(time_start) for time_start in responses['time_bin_start']]}]
        # select time
        time_responses = []
        for time_bin in tqdm(self.recording_time_bins, desc='CORnet-time to recording time'):
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


def vonecornet_s_brainmodel():
    # map region -> (time_start, time_step_size, timesteps)
    time_mappings = {
        'V1': (50, 100, 1),
        'V2': (70, 100, 2),
        # 'V2': (20, 50, 2),  # MS: This follows from the movshon anesthesized-monkey recordings, so might not hold up
        'V4': (90, 50, 4),
        'IT': (100, 100, 2),
    }
    return VOneCORnetCommitment(identifier='CORnet-S', activations_model=vonecornet('cornets'),
                                layers=['vone_block.output-t0'] + [f'model.{area}.output-t{timestep}'
                                                                   for area, timesteps in
                                                                   [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                                                   for timestep in timesteps] +
                                       ['model.decoder.avgpool-t0'], time_mapping=_build_time_mappings(time_mappings))


class VOneCORnetBrainPool(UniqueKeyDict):
    def __init__(self):
        super(VOneCORnetBrainPool, self).__init__(reload=True)

        model_pool = {
            'vonecornets': LazyLoad(vonecornet_s_brainmodel),
        }

        self._accessed_brain_models = []

        for identifier, brain_model in model_pool.items():
            self[identifier] = brain_model


vonecornet_brain_pool = VOneCORnetBrainPool()
