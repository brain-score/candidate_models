import logging

import numpy as np

from neurality import models
from neurality.assemblies import load_neural_benchmark, load_stimulus_set
from neurality.models import model_activations, model_graph
from neurality.models.graph import combine_graph, cut_graph
from neurality.storage import get_function_identifier, store

logger = logging.getLogger(__name__)


class Defaults(object):
    neural_data = 'dicarlo.majaj2015'


@store()
def score_model(model, layers, neural_data=Defaults.neural_data, model_weights=models.Defaults.model_weights):
    physiology_score = score_physiology(model=model, layers=layers, neural_data=neural_data,
                                        model_weights=model_weights)
    anatomy_score = score_anatomy(model, physiology_score.mapping)
    return [physiology_score, anatomy_score]


def score_physiology(model, layers=None, neural_data=Defaults.neural_data, model_weights=models.Defaults.model_weights):
    """
    :param str model:
    :param [str]|None layers: layers to score or None to use all layers present in the model activations
    :param str neural_data:
    :param str model_weights:
    :return: PhysiologyScore
    """
    logger.info('Computing activations')
    model_assembly = model_activations(model=model, model_weights=model_weights, layers=layers,
                                       stimulus_set=neural_data)

    # run by layer so that we can store individual scores
    layers = layers or np.unique(model_assembly['layer'])
    logger.info('Scoring layers: {}'.format(layers))
    layer_scores = []
    for layer_name in layers:
        layer_activations = model_assembly.sel(layer=layer_name).stack(neuroid=('neuroid_id',))
        score = score_physiology_layer(model=model, layer=layer_name, activations=layer_activations,
                                       neural_data=neural_data)
        score.name = layer_name
        layer_scores.append(score)
    max_score = max(layer_scores, key=lambda score: score.center)
    max_score.explanation = layer_scores
    return max_score


@store(filename_ignore=['activations'])
def score_physiology_layer(model, layer, activations, neural_data, metric_name='neural_fit'):
    benchmark = load_neural_benchmark(assembly_name=neural_data, metric_name=metric_name)
    score = benchmark(activations)
    return score


def score_anatomy(model, region_layers):
    graph = model_graph(model, layers=list(region_layers.values()))
    graph = combine_graph(graph, region_layers)
    graph = cut_graph(graph, keep_nodes=relevant_regions)
    benchmark = load_neural_benchmark(assembly_name='ventral_stream', metric_name='edge_ratio')
    score = benchmark(graph)
    return score
