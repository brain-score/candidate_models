import logging

import brainscore
from brainscore.benchmarks import benchmark_pool
from candidate_models.model_commitments import brain_translated_pool
from result_caching import store

_logger = logging.getLogger(__name__)


@store(identifier_ignore=['model', 'benchmark'])
def score_model(model_identifier, benchmark_identifier, model, benchmark=None):
    # model_identifier variable is not unused, the result caching component uses it to identify the cached results
    assert model is not None
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]
    _logger.debug("scoring model")
    score = benchmark(model)
    score.attrs['model_identifier'] = model_identifier
    score.attrs['benchmark_identifier'] = benchmark_identifier
    return score


def get_activations(model, layers, stimulus_set):
    stimuli = brainscore.get_stimulus_set(stimulus_set)
    return model(stimuli, layers=layers)
