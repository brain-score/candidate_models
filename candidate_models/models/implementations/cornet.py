import importlib
import os
import re
from collections import defaultdict

import torch
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
        'S': 'cornet_s_epoch43.pth.tar'
    }

    def __init__(self, *args, cornet_type, **kwargs):
        self._cornet_type = cornet_type
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
