from brainio_base.assemblies import NeuroidAssembly, walk_coords, merge_data_arrays
from brainscore.model_interface import BrainModel
from model_tools.brain_transformation import LayerSelection


class LayersAsTimesteps(BrainModel):
    def __init__(self, identifier, activations_model, layers, behavioral_readout_layer=None):
        self.identifier = identifier
        self.layers = layers
        self.region_assemblies = {}
        self.region_layer_map = {}
        self.recording_target = None
        self.activations_model = activations_model
        self.do_behavior = False

    def look_at(self, stimuli):
        assert self.recording_target == 'IT'
        v4_layer, it_layer = self.region_layer_map['V4'], self.region_layer_map['IT']
        layers, add = [], False
        for layer in self.layers:
            if layer == v4_layer:
                add = True
                continue
            if layer == it_layer:
                break
            if add:
                layers.append(layer)

        activations = self.activations_model(stimuli, layers=layers)
        activations['region'] = 'neuroid', [self.recording_target] * len(activations['neuroid'])
        time_binned_activations = []
        time_step_size = 10
        for layer, time_bin_start in zip(layers, range(70, 70 + len(layers) * time_step_size, time_step_size)):
            layer_activations = activations.sel(layer=layer)
            layer_activations = NeuroidAssembly([layer_activations.values], coords={
                **{coord: (dims, values) for coord, dims, values in walk_coords(layer_activations)
                   if coord not in ['neuroid_id']},
                **{'time_bin_start': ('time_bin', [time_bin_start]),
                   'time_bin_end': ('time_bin', [time_bin_start + time_step_size])}
            }, dims=['time_bin'] + list(layer_activations.dims))
            time_binned_activations.append(layer_activations)
        activations = merge_data_arrays(time_binned_activations)
        return activations

    def commit_region(self, region, assembly, assembly_stratification=None):
        self.region_assemblies[region] = (assembly, assembly_stratification)  # lazy, only run when actually needed

    def do_commit_region(self, region):
        layer_selection = LayerSelection(model_identifier=self.identifier,
                                         activations_model=self.activations_model, layers=self.layers)
        assembly, assembly_stratification = self.region_assemblies[region]
        best_layer = layer_selection(assembly, assembly_stratification=assembly_stratification)
        self.commit(region, best_layer)

    def commit(self, region, layer):
        self.region_layer_map[region] = layer

    def start_recording(self, recording_target, time_bins):
        if recording_target not in self.region_layer_map:  # not yet committed
            self.do_commit_region(recording_target)
            self.do_commit_region('V4')
        self.recording_target, self.recording_time_bins = recording_target, time_bins
