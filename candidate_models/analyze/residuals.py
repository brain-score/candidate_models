import logging
import sys

from caching import store_xarray
from brainscore.assemblies import walk_coords, DataAssembly
from brainscore.benchmarks import DicarloMajaj2015
from brainscore.metrics.ceiling import SplitNoCeiling
from brainscore.metrics.neural_fit import NeuralFit
from candidate_models import Defaults, model_layers, model_multi_activations

_logger = logging.getLogger(__name__)


class ResidualNeuralFit(NeuralFit):
    def compare_prediction(self, prediction, target):
        residuals = target.values - prediction.values
        # re-package; excluding region since it forms another dimension in CartesianProduct
        coords = {name: (dims, values) for name, dims, values in walk_coords(target) if name != 'region'}
        residuals = DataAssembly(residuals, coords=coords, dims=target.dims)
        return residuals

    def aggregate(self, residuals):
        return residuals


class ResidualBenchmark(DicarloMajaj2015):
    def __init__(self):
        super(ResidualBenchmark, self).__init__()
        self._ceiling = SplitNoCeiling()
        self._metric = ResidualNeuralFit()


def run(model, layers=None, neural_data=Defaults.neural_data):
    layers = layers or model_layers[model]
    return _run(model=model, layers=layers, neural_data=neural_data)


@store_xarray(combine_fields={'layers': 'layer'}, identifier_ignore=['layers'])
def _run(model, layers, neural_data=Defaults.neural_data):
    assert neural_data == 'dicarlo.Majaj2015'
    _logger.info('Computing activations')
    model_assembly = model_multi_activations(model=model, multi_layers=layers, stimulus_set=neural_data)
    _logger.info('Loading data')
    benchmark = ResidualBenchmark()
    _logger.info('Computing residuals')
    residuals = benchmark(model_assembly, source_splits=['layer'])
    residuals = residuals.aggregation.sel(aggregation='center')
    return residuals


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    run('alexnet')
