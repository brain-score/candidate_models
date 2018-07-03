import brainscore
from brainscore.benchmarks import build as build_benchmark


def load_neural_benchmark(assembly_name, metric_name, ceiling_name=None, target_splits=()):
    return build_benchmark(assembly_name=assembly_name, metric_name=metric_name, ceiling_name=ceiling_name,
                           target_splits=target_splits)


def load_stimulus_set(assembly_name):
    assembly = brainscore.get_assembly(assembly_name)
    stimulus_set = assembly.attrs['stimulus_set']
    return stimulus_set
