import mkgu

from neurality.storage import cache


@cache()
def load_neural_benchmark(data_name, metric_name):
    return mkgu.metrics.benchmarks.load(data_name=data_name, metric_name=metric_name)


def load_stimuli(name):
    stimuli = mkgu.get_stimuli(name).load()
    return stimuli
