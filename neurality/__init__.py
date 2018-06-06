import logging

from mkgu.metrics import CartesianProduct, CrossValidation
from neurality import models
from neurality.assemblies import load_neural_benchmark, load_stimulus_set
from neurality.models import model_activations, model_multi_activations, combine_layers_xarray, split_layers_xarray
from neurality.models.graph import combine_graph, cut_graph
from neurality.models.implementations import Defaults as DeepModelDefaults
from neurality.models.implementations import model_layers
from neurality.storage import store, store_xarray

logger = logging.getLogger(__name__)


class Defaults(object):
    neural_data = 'dicarlo.Majaj2015'
    metric_name = 'neural_fit'


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


@store_xarray(identifier_ignore=['layers', 'image_size'], combine_fields={'layers': 'layer'},
              map_field_values=_combine_layers, map_field_values_inverse=_un_combine_layers,
              sub_fields=True)
def score_physiology(model, layers=None,
                     weights=DeepModelDefaults.weights,
                     pca_components=DeepModelDefaults.pca_components, image_size=DeepModelDefaults.image_size,
                     neural_data=Defaults.neural_data, metric_name=Defaults.metric_name):
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
    layers = layers or model_layers[model]
    logger.info('Computing activations')
    model_assembly = model_multi_activations(model=model, weights=weights, multi_layers=layers,
                                             pca_components=pca_components, image_size=image_size,
                                             stimulus_set=neural_data)
    logger.info('Loading benchmark')
    benchmark = load_neural_benchmark(
        assembly_name=neural_data, metric_name=metric_name, metric_kwargs=dict(transformations=[
            CartesianProduct(dividing_coord_names_source=['layer'], dividing_coord_names_target=['region']),
            CrossValidation()]))
    logger.info('Scoring activations')
    score = benchmark(model_assembly)
    return score


def score_anatomy(model, region_layers):
    graph = model_graph(model, layers=list(region_layers.values()))
    graph = combine_graph(graph, region_layers)
    graph = cut_graph(graph, keep_nodes=relevant_regions)
    benchmark = load_neural_benchmark(assembly_name='ventral_stream', metric_name='edge_ratio')
    score = benchmark(graph)
    return score
