import re
from collections import defaultdict

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
