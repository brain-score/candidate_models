import functools
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from mkgu.assemblies import NeuroidAssembly, merge_data_arrays
from mkgu.metrics import subset, get_modified_coords
from .implementations import Defaults as DeepModelDefaults
from .implementations.keras import KerasModel
from .implementations.pytorch import PytorchModel
from .implementations.tensorflow_slim import TensorflowSlimModel
from neurality.storage import store_xarray

_logger = logging.getLogger(__name__)


class Defaults(object):
    stimulus_set = 'dicarlo.Majaj2015'


def model_multi_activations(model, multi_layers, stimulus_set=Defaults.stimulus_set,
                            weights=DeepModelDefaults.weights, image_size=DeepModelDefaults.image_size,
                            pca_components=DeepModelDefaults.pca_components, batch_size=DeepModelDefaults.batch_size):
    single_layers = []
    for layers in multi_layers:
        if isinstance(layers, str):
            single_layers.append(layers)
        else:
            for layer in layers:
                single_layers.append(layer)
    single_layers = list(set(single_layers))
    single_layer_activations = model_activations(model, single_layers, stimulus_set, weights=weights,
                                                 image_size=image_size, pca_components=pca_components,
                                                 batch_size=batch_size)

    multi_layer_activations = []
    for layers in multi_layers:
        if isinstance(layers, str):
            layers = [layers]
        layers_target = xr.DataArray(np.full(len(layers), np.nan), coords={'layer': layers}, dims=['layer'])
        layers_target = layers_target.stack(neuroid=['layer'])
        layers_activations = subset(single_layer_activations, layers_target, dims_must_match=False)

        # at this point, layers_activations are concatenated across layers
        # BUT they will be disconnected again later due to layer being an adjacent coordinate.
        # we set `layer` to the concatenated coords and keep the original `layer` in another coord.
        noncombined_layers = layers_activations['layer'].dims, layers_activations['layer'].values

        def modify_coord(name, dims, values):
            # we can't build a list here because xarray won't allow that later on. instead, combine with string join
            return name, (dims, values if name != 'layer' else np.repeat(combine_layers_xarray(layers), len(values)))

        coords = get_modified_coords(layers_activations, modify_coord)
        coords['noncombined_layer'] = noncombined_layers
        layers_activations = NeuroidAssembly(layers_activations.values, coords=coords, dims=layers_activations.dims)

        multi_layer_activations.append(layers_activations)
    return merge_data_arrays(multi_layer_activations)


def combine_layers_xarray(layers):
    return ",".join(layers)


def split_layers_xarray(layers):
    return layers.split(",")


def package_stimulus_coords(assembly, stimulus_set):
    stimulus_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
    assert all(assembly['stimulus_path'] == stimulus_paths)
    assembly['stimulus_path'] = stimulus_set['image_id'].values
    assembly = assembly.rename({'stimulus_path': 'image_id'})
    for column in stimulus_set.columns:
        assembly[column] = 'image_id', stimulus_set[column].values
    assembly = assembly.stack(presentation=('image_id',))
    return assembly


@store_xarray(identifier_ignore=['layers', 'batch_size', 'pca_batch_size'], combine_fields={'layers': 'layer'})
def model_activations(model, layers, stimulus_set=Defaults.stimulus_set, weights=DeepModelDefaults.weights,
                      image_size=DeepModelDefaults.image_size, pca_components=DeepModelDefaults.pca_components,
                      batch_size=DeepModelDefaults.batch_size):
    from neurality import load_stimulus_set
    _logger.info('Loading stimuli')
    stimulus_set = load_stimulus_set(stimulus_set)
    stimuli_paths = list(map(stimulus_set.get_image, stimulus_set['image_id']))

    _logger.info('Creating model')
    model = create_model(model=model, weights=weights, batch_size=batch_size, image_size=image_size)
    _logger.debug(str(model))

    _logger.info('Computing activations')
    assembly = model.get_activations(stimuli_paths=stimuli_paths, layers=layers, pca_components=pca_components)
    assembly = package_stimulus_coords(assembly, stimulus_set)
    return assembly


def create_model(model, *args, **kwargs):
    return models[model](*args, **kwargs)


def load_model_definitions():
    models = {}
    models_meta = pd.read_csv(os.path.join(os.path.dirname(__file__), 'implementations', 'models.csv'),
                              keep_default_na=False)
    for _, row in models_meta.iterrows():
        framework = row['framework']
        if not framework:  # basenet
            continue
        framework = {'keras': KerasModel, 'pytorch': PytorchModel, 'slim': TensorflowSlimModel}[framework]
        model = row['model']
        models[model] = functools.partial(framework, model_name=model)
    return models


models = load_model_definitions()
