import logging

import functools
from tqdm import tqdm

from brainscore.benchmarks import benchmark_pool
from brainscore.metrics import Score
from candidate_models.model_commitments import mapping_model_pool
from result_caching import store

_logger = logging.getLogger(__name__)


@store(identifier_ignore=['model', 'benchmark'])
def map_and_score_model(model_identifier, benchmark_identifier, model=None, benchmark=None):
    if model is None:
        _logger.debug("retrieving model")
        model = mapping_model_pool[model_identifier]
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]
    _logger.debug("mapping model")
    mapped_model = model.map(benchmark.training_benchmark, benchmark.validation_benchmark)
    _logger.debug("scoring mapped model")
    score = benchmark(mapped_model)
    return score


@store(identifier_ignore=['model', 'benchmark'])
def score_model(model_identifier, benchmark_identifier, model, benchmark=None):
    assert model is not None
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]
    _logger.debug("scoring model")
    score = benchmark(model)
    return score


@store(identifier_ignore=['model', 'benchmark', 'layers'])
def score_layers(model_identifier, benchmark_identifier, model, layers, benchmark=None):
    assert model is not None
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]

    def get_activations(stimulus_set, layer):
        # for efficiency, we compute activations for all the layers (which will be stored on disk),
        # then select only the current layer
        all_layers = model.from_stimulus_set(stimulus_set, layers=layers)
        activations = all_layers.sel(layer=layer)
        return activations.stack(neuroid=['neuroid_id'])

    layer_scores = []
    for layer in tqdm(layers):
        _logger.debug(f"scoring {model_identifier}, layer {layer}")
        layer_activations = functools.partial(get_activations, layer=layer)
        score = benchmark(layer_activations)
        score = score.expand_dims('layer')
        score['layer'] = [layer]
        layer_scores.append(score)
    score = Score.merge(*layer_scores)
    return score
