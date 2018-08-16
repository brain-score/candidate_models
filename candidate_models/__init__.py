import logging
import os

import caching
from brainscore import benchmarks
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
    metric_name = 'pls_fit'
    target_splits = ('region',)


def score_model(model, layers, weights=DeepModelDefaults.weights,
                pca_components=DeepModelDefaults.pca_components, image_size=DeepModelDefaults.image_size,
                neural_data=Defaults.neural_data):
    physiology_score = score_physiology(model=model, layers=layers, weights=weights,
                                        pca_components=pca_components, image_size=image_size,
                                        neural_data=neural_data)
    return physiology_score


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
                     neural_data=Defaults.neural_data, return_unceiled=False):
    """
    :param str model:
    :param [str]|None layers: layers to score or None to use all layers present in the model activations
    :param str weights:
    :param int pca_components:
    :param str neural_data:
    :param int image_size:
    :return: PhysiologyScore
    """
    # this method is just a wrapper function around _score_physiology
    # so that we can properly handle default values for `layers`.
    layers = layers or model_layers[model]
    logger.info('Loading benchmark')
    benchmark = benchmarks.load(neural_data)
    logger.info('Computing activations')
    model_assembly = model_multi_activations(model=model, weights=weights, multi_layers=layers,
                                             pca_components=pca_components, image_size=image_size,
                                             stimulus_set=benchmark.stimulus_set_name)
    logger.info('Scoring activations')
    ceiled_score, unceiled_score = benchmark(model_assembly, transformation_kwargs=dict(
        cartesian_product_kwargs=dict(dividing_coord_names_source=['layer'])), return_unceiled=True)
    if return_unceiled:
        return ceiled_score, unceiled_score
    return ceiled_score


def score_anatomy(model, region_layers):
    graph = model_graph(model, layers=list(region_layers.values()))
    graph = combine_graph(graph, region_layers)
    graph = cut_graph(graph, keep_nodes=relevant_regions)
    benchmark = load_neural_benchmark(assembly_name='ventral_stream', metric_name='edge_ratio')
    score = benchmark(graph)
    return score
