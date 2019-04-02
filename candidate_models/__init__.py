import logging

from brainscore.benchmarks import benchmark_pool
from brainscore.benchmarks.loaders import load_assembly
from candidate_models.model_commitments import brain_translated_pool
from model_tools.brain_transformation import LayerScores
from result_caching import store

_logger = logging.getLogger(__name__)


@store(identifier_ignore=['model', 'benchmark'])
def map_and_score_model(model_identifier, benchmark_identifier, model=None, benchmark=None):
    if model is None:
        _logger.debug("retrieving model")
        model = brain_translated_pool[model_identifier]
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


def score_layers(model_identifier, benchmark_identifier, model, layers, benchmark=None):
    assert model is not None
    if benchmark is None:
        _logger.debug("retrieving benchmark")
        benchmark = benchmark_pool[benchmark_identifier]

    scorer = LayerScores(model_identifier=model_identifier, activations_model=model)
    scores = scorer(benchmark=benchmark, benchmark_identifier=benchmark_identifier, layers=layers)
    return scores


def get_activations(model, layers, assembly_identifier):
    assembly = load_assembly(assembly_identifier)
    stimuli = assembly.stimulus_set
    return model(stimuli, layers=layers)
