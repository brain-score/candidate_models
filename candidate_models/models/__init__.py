import functools
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from result_caching import store_xarray
from brainscore.assemblies import NeuroidAssembly, merge_data_arrays
from brainscore.metrics.transformations import subset
from brainscore.metrics.utils import get_modified_coords
from .implementations import Defaults as DeepModelDefaults, DeepModel
from .implementations.keras import KerasModel
from .implementations.pytorch import PytorchPredefinedModel
from .implementations.tensorflow_slim import TensorflowSlimPredefinedModel

_logger = logging.getLogger(__name__)


class Defaults(object):
    stimulus_set = 'dicarlo.hvm'


def model_multi_activations(model, multi_layers, stimulus_set=Defaults.stimulus_set,
                            model_identifier=None, weights=DeepModelDefaults.weights,
                            image_size=DeepModelDefaults.image_size, pca_components=DeepModelDefaults.pca_components,
                            batch_size=DeepModelDefaults.batch_size):
    """
    :param model:
    :param multi_layers:
    :param stimulus_set:
    :param model_identifier: optional string for the model name.
        only required when model is not a string pointing to a saved model.
    :param weights:
    :param image_size:
    :param pca_components:
    :param batch_size:
    :return:
    """
    single_layers = []
    for layers in multi_layers:
        if isinstance(layers, str):
            single_layers.append(layers)
        else:
            for layer in layers:
                single_layers.append(layer)
    # remove duplicates, restore ordering
    single_layers = list(sorted(set(single_layers), key=single_layers.index))
    single_layer_activations = model_activations(model, single_layers, stimulus_set,
                                                 model_identifier=model_identifier, weights=weights,
                                                 image_size=image_size, pca_components=pca_components,
                                                 batch_size=batch_size)
    multi_layer_necessary = any(not isinstance(layer, str) for layer in multi_layers)
    if not multi_layer_necessary:
        return single_layer_activations

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
    multi_layer_activations = merge_data_arrays(multi_layer_activations)
    multi_layer_activations.name = single_layer_activations.name
    return multi_layer_activations


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


def model_activations(model, layers, stimulus_set=Defaults.stimulus_set, model_identifier=None,
                      weights=DeepModelDefaults.weights,
                      image_size=None, pca_components=DeepModelDefaults.pca_components,
                      batch_size=DeepModelDefaults.batch_size):
    if isinstance(model, str):
        assert model_identifier is None or model_identifier == model, \
            f"model identifier {model_identifier} does not match model string {model}"
        model_identifier = model
        _logger.info('Will use pre-defined model')
        model_ctr = functools.partial(create_model, model=model)
    else:
        assert model_identifier is not None, "need model_identifier to save activations"
        _logger.debug('Will use passed model constructor')
        model_ctr = model
    image_size = image_size or infer_image_size(model_identifier)

    return _model_activations(model_ctr=model_ctr, layers=layers, stimulus_set=stimulus_set,
                              model_identifier=model_identifier, weights=weights,
                              image_size=image_size, pca_components=pca_components, batch_size=batch_size)


@store_xarray(identifier_ignore=['model_ctr', 'layers', 'batch_size', 'pca_batch_size'],
              combine_fields={'layers': 'layer'})
def _model_activations(model_identifier, model_ctr, layers, stimulus_set, weights,
                       image_size, pca_components, batch_size):
    from candidate_models.assemblies import load_stimulus_set
    _logger.info('Loading stimuli')
    stimulus_set = load_stimulus_set(stimulus_set)
    stimuli_paths = list(map(stimulus_set.get_image, stimulus_set['image_id']))

    _logger.info(f'Creating model {model_identifier}')
    model = model_ctr(weights=weights, batch_size=batch_size, image_size=image_size)
    _logger.debug(str(model))

    _logger.info('Retrieving activations')
    assembly = model.get_activations(stimuli_paths=stimuli_paths, layers=layers, pca_components=pca_components)
    assembly = package_stimulus_coords(assembly, stimulus_set)
    assembly.name = model_identifier
    return assembly


def create_model(model, *args, **kwargs):
    return models[model](*args, **kwargs)


def cornet(*args, cornet_type, **kwargs):
    from candidate_models.models.implementations.cornet import CORNetWrapper
    return CORNetWrapper(*args, cornet_type=cornet_type, **kwargs)


def load_model_definitions():
    models = {}
    for _, row in models_meta.iterrows():
        framework = row['framework']
        if not framework:  # basenet
            continue
        framework = {'keras': KerasModel,
                     'pytorch': PytorchPredefinedModel,
                     'slim': TensorflowSlimPredefinedModel}[framework]
        model = row['model']
        models[model] = functools.partial(framework, model_name=model)
    for cornet_type in ['Z', 'R', 'S', 'R2']:
        models[f'cornet_{cornet_type.lower()}'] = functools.partial(cornet, cornet_type=cornet_type)
    return models


def load_model_meta():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'implementations', 'models.csv'),
                       keep_default_na=False)


models_meta = load_model_meta()
models = load_model_definitions()


def infer_image_size(model):
    meta = models_meta[models_meta['model'] == model]
    image_size = meta['image_size']
    if len(image_size) != 1 or np.isnan(image_size.values[0]):
        raise ValueError("Could not lookup image size")
    return int(image_size.values[0])
