import logging

from neurality import models
from neurality.assemblies import load_neural_benchmark, load_stimulus_set
from neurality.models import model_activations, model_graph, model_multi_activations, \
    combine_layers_xarray, split_layers_xarray
from neurality.models.graph import combine_graph, cut_graph
from neurality.models.implementations import model_layers
from neurality.storage import store, store_xarray

logger = logging.getLogger(__name__)


class Defaults(object):
    neural_data = 'dicarlo.majaj2015'


def score_model(model, layers, neural_data=Defaults.neural_data, model_weights=models.Defaults.model_weights):
    physiology_score = score_physiology(model=model, layers=layers, neural_data=neural_data,
                                        model_weights=model_weights)
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


@store_xarray(identifier_ignore=['layers', 'image_size'], combine_fields={'layers': 'layer'},
              map_field_values=_combine_layers, map_field_values_inverse=_un_combine_layers,
              sub_fields=True)
def score_physiology(model, layers=None,
                     model_weights=models.Defaults.model_weights,
                     pca_components=models.Defaults.pca_components, image_size=models.Defaults.image_size,
                     neural_data=Defaults.neural_data, metric_name=Defaults.metric_name):
    """
    :param str model:
    :param [str]|None layers: layers to score or None to use all layers present in the model activations
    :param str model_weights:
    :param int pca_components:
    :param str neural_data:
    :param str metric_name:
    :param int image_size:
    :return: PhysiologyScore
    """
    layers = layers or model_layers[model]
    logger.info('Computing activations')
    model_assembly = model_multi_activations(model=model, model_weights=model_weights, multi_layers=layers,
                                             pca_components=pca_components, image_size=image_size,
                                             stimulus_set=neural_data)
    logger.info('Loading benchmark')
    benchmark = load_neural_benchmark(assembly_name=neural_data, metric_name=metric_name)
    logger.info('Scoring activations')
    score = benchmark(model_assembly, metric_kwargs=dict(similarity_kwargs=dict(
        additional_adjacent_coords=['region', 'layer'])))
    return score


def score_anatomy(model, region_layers):
    graph = model_graph(model, layers=list(region_layers.values()))
    graph = combine_graph(graph, region_layers)
    graph = cut_graph(graph, keep_nodes=relevant_regions)
    benchmark = load_neural_benchmark(assembly_name='ventral_stream', metric_name='edge_ratio')
    score = benchmark(graph)
    return score
