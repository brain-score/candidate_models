import importlib
import logging
import os
from collections import defaultdict

import functools
import re
import torch
from torch.nn import Module

from brainio_base.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from candidate_models import s3
from candidate_models.base_models.cornet.cornet_r2 import fix_state_dict_naming as fix_r2_state_dict_naming
from model_tools.activations.pytorch import PytorchWrapper

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
        from .cornet_r2 import CORNetR2
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
    if cornet_type.lower() == 'r2':
        checkpoint['state_dict'] = fix_r2_state_dict_naming(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module  # unwrap

    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = TemporalPytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
