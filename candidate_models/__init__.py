import logging

from brainscore import benchmarks
from candidate_models import models
from candidate_models.assemblies import load_neural_benchmark, load_stimulus_set
from candidate_models.models import model_activations, model_multi_activations, combine_layers_xarray, \
    split_layers_xarray
from candidate_models.models.graph import combine_graph, cut_graph
from candidate_models.models.implementations import Defaults as DeepModelDefaults
from candidate_models.models.implementations import model_layers

logger = logging.getLogger(__name__)


class Defaults(object):
    benchmark = 'brain-score'


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


def score_model(model, model_identifier=None, layers=None,
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
    :return: Score
    """
    if layers is None:
        assert isinstance(model, str), "need either known model string or list of layers"
        layers = model_layers[model]

    assert model_identifier is not None or isinstance(model, str), "need either known model string or model_identifier"
    model_name = model_identifier if model_identifier is not None else model

    logger.info('Loading benchmark')
    benchmark = benchmarks.load(benchmark)

    # package model assembly in lazily-loaded promise
    # so that we don't need to have activations stored locally when we just want to look at scores
    def _compute_activations():
        logger.info('Computing activations')
        model_assembly = model_multi_activations(model=model, model_identifier=model_identifier,
                                                 weights=weights, multi_layers=layers,
                                                 pca_components=pca_components, image_size=image_size,
                                                 stimulus_set=benchmark.stimulus_set_name)
        return model_assembly

    promise = AssemblyPromise(name=model_name, load_fnc=_compute_activations)

    logger.info(f'Scoring {model_name}')
    score = benchmark(promise, transformation_kwargs=dict(
        cartesian_product_kwargs=dict(dividing_coord_names_source=['layer'])), return_ceiled=return_ceiled)
    return score
