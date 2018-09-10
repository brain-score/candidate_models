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
    benchmark = 'dicarlo.Majaj2015'


def score_model(model, layers=None, weights=DeepModelDefaults.weights,
                pca_components=DeepModelDefaults.pca_components, image_size=DeepModelDefaults.image_size,
                benchmark=Defaults.benchmark):
    physiology_score = score_physiology(model=model, layers=layers, weights=weights,
                                        pca_components=pca_components, image_size=image_size,
                                        benchmark=benchmark)
    return physiology_score


class AssemblyPromise(object):
    def __init__(self, name, load_fnc):
        self.name = name
        self._load = load_fnc
        self.values = None

    def __getattr__(self, item):
        if item == 'name':
            return super(AssemblyPromise, self).__getattr__(item)
        self._ensure_loaded()
        return getattr(self.values, item)

    def __getitem__(self, item):
        self._ensure_loaded()
        return self.values[item]

    def _ensure_loaded(self):
        if self.values is None:
            self.values = self._load()


def score_physiology(model, layers=None,
                     weights=DeepModelDefaults.weights,
                     pca_components=DeepModelDefaults.pca_components, image_size=DeepModelDefaults.image_size,
                     benchmark=Defaults.benchmark, return_ceiled=False):
    """
    :param str model:
    :param [str]|None layers: layers to score or None to use all layers present in the model activations
    :param str weights:
    :param int pca_components:
    :param str benchmark:
    :param int image_size:
    :return: PhysiologyScore
    """
    if layers is None:
        assert isinstance(model, str), "need either known model string or list of layers"
        layers = model_layers[model]
    logger.info('Loading benchmark')
    benchmark = benchmarks.load(benchmark)

    def _compute_activations():
        logger.info('Computing activations')
        model_assembly = model_multi_activations(model=model, weights=weights, multi_layers=layers,
                                                 pca_components=pca_components, image_size=image_size,
                                                 stimulus_set=benchmark.stimulus_set_name)
        return model_assembly

    promise = AssemblyPromise(model, _compute_activations)

    logger.info(f'Scoring {model}')
    score = benchmark(promise, transformation_kwargs=dict(
        cartesian_product_kwargs=dict(dividing_coord_names_source=['layer'])), return_ceiled=return_ceiled)
    return score


def score_anatomy(model, region_layers):
    graph = model_graph(model, layers=list(region_layers.values()))
    graph = combine_graph(graph, region_layers)
    graph = cut_graph(graph, keep_nodes=relevant_regions)
    benchmark = load_neural_benchmark(assembly_name='ventral_stream', metric_name='edge_ratio')
    score = benchmark(graph)
    return score


def _combine_layers(key, value):
    if key != 'layer':
        return value
    return [combine_layers_xarray(layers) if not isinstance(layers, str) else layers for layers in value]


def _un_combine_layers(key, value):
    if key != 'layer':
        return value
    return [split_layers_xarray(layers) if ',' in layers else layers for layers in value]
