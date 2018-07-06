import logging
import os

import caching
from caching import store_xarray
from candidate_models import models
from candidate_models.assemblies import load_neural_benchmark, load_stimulus_set
from candidate_models.models import model_activations, model_multi_activations, combine_layers_xarray, \
    split_layers_xarray
from candidate_models.models.graph import combine_graph, cut_graph
from candidate_models.models.implementations import Defaults as DeepModelDefaults
from candidate_models.models.implementations import model_layers

logger = logging.getLogger(__name__)
caching.store.configure_storagedir(os.path.join(os.path.dirname(__file__), '..', 'output'))


class Defaults(object):
    neural_data = 'dicarlo.Majaj2015'
    metric_name = 'neural_fit'
    target_splits = ('region',)


def score_model(model, layers, weights=DeepModelDefaults.weights,
                pca_components=DeepModelDefaults.pca_components, image_size=DeepModelDefaults.image_size,
                neural_data=Defaults.neural_data, metric_name=Defaults.metric_name):
    physiology_score = score_physiology(model=model, layers=layers, weights=weights,
                                        pca_components=pca_components, image_size=image_size,
                                        neural_data=neural_data, metric_name=metric_name)
    anatomy_score = score_anatomy(model, physiology_score.mapping)
    return [physiology_score, anatomy_score]


def _combine_layers(key, value):
    if key != 'layer':
        return value
    return [combine_layers_xarray(layers) if not isinstance(layers, str) else layers for layers in value]


def _un_combine_layers(key, value):
    if key != 'layer':
        return value
    return [split_layers_xarray(layers) if ',' in layers else layers for layers in value]


def score_physiology(model, layers=None,
                     weights=DeepModelDefaults.weights,
                     pca_components=DeepModelDefaults.pca_components, image_size=DeepModelDefaults.image_size,
                     neural_data=Defaults.neural_data, target_splits=Defaults.target_splits,
                     metric_name=Defaults.metric_name, return_unceiled=False):
    """
    :param str model:
    :param [str]|None layers: layers to score or None to use all layers present in the model activations
    :param str weights:
    :param int pca_components:
    :param str neural_data:
    :param str metric_name:
    :param int image_size:
    :return: PhysiologyScore
    """
    # this method is just a wrapper function around _score_physiology
    # so that we can properly handle default values for `layers`.
    layers = layers or model_layers[model]
    ceiled_score, unceiled_score = _score_physiology(model=model, layers=layers, weights=weights,
                                                     pca_components=pca_components, image_size=image_size,
                                                     neural_data=neural_data, target_splits=target_splits,
                                                     metric_name=metric_name)
    if return_unceiled:
        return ceiled_score, unceiled_score
    return ceiled_score


@store_xarray(identifier_ignore=['layers', 'image_size', 'metric_kwargs'], combine_fields={'layers': 'layer'},
              map_field_values=_combine_layers, map_field_values_inverse=_un_combine_layers,
              sub_fields=True)
def _score_physiology(model, layers,
                      weights=DeepModelDefaults.weights,
                      pca_components=DeepModelDefaults.pca_components, image_size=DeepModelDefaults.image_size,
                      neural_data=Defaults.neural_data, target_splits=Defaults.target_splits,
                      metric_name=Defaults.metric_name):
    layers = layers or model_layers[model]
    logger.info('Computing activations')
    model_assembly = model_multi_activations(model=model, weights=weights, multi_layers=layers,
                                             pca_components=pca_components, image_size=image_size,
                                             stimulus_set=neural_data)
    logger.info('Loading benchmark')
    benchmark = load_neural_benchmark(assembly_name=neural_data, metric_name=metric_name, target_splits=target_splits)
    logger.info('Scoring activations')
    ceiled_score, unceiled_score = benchmark(model_assembly, source_splits=['layer'], return_unceiled=True)
    return ceiled_score, unceiled_score


def score_anatomy(model, region_layers):
    graph = model_graph(model, layers=list(region_layers.values()))
    graph = combine_graph(graph, region_layers)
    graph = cut_graph(graph, keep_nodes=relevant_regions)
    benchmark = load_neural_benchmark(assembly_name='ventral_stream', metric_name='edge_ratio')
    score = benchmark(graph)
    return score
