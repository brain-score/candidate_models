import logging

from brainscore.benchmarks import benchmark_pool
from candidate_models.mapping_models import mapping_model_pool
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
