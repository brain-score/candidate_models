import mkgu
from mkgu.benchmarks import build as build_benchmark
from mkgu.stimuli import StimulusSet

from candidate_models.storage import cache
from mkgu.benchmarks import load as load_benchmark


def load_neural_benchmark(assembly_name, metric_name, ceiling_name=None, target_splits=()):
    return build_benchmark(assembly_name=assembly_name, metric_name=metric_name, ceiling_name=ceiling_name,
                           target_splits=target_splits)


def load_stimulus_set(assembly_name):
    assembly = mkgu.get_assembly(assembly_name)
    stimulus_set = assembly.attrs['stimulus_set']
    return stimulus_set
