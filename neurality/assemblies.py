import mkgu

from neurality.storage import cache
from mkgu.benchmarks import load as load_benchmark


@cache()
def load_neural_benchmark(assembly_name, metric_name):
    return load_benchmark(data_name=assembly_name, metric_name=metric_name)


def load_stimulus_set(assembly_name):
    assembly = mkgu.get_assembly(assembly_name)
    stimulus_set = assembly.attrs['stimulus_set']
    return stimulus_set
