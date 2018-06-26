import logging
import os
import sys

import numpy as np
import scipy
from matplotlib import pyplot

from mkgu.assemblies import merge_data_arrays
from mkgu.metrics.ceiling import ceilings
from neurality import load_neural_benchmark
from neurality.storage import store, cache

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

_logger = logging.getLogger(__name__)


def plot(train_sizes=[0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], data='dicarlo.Majaj2015', metric='neural_fit',
         ceiling='splitrep'):
    # compute
    ceilings = []
    for train_size in train_sizes:
        score = compute_ceilings(data, metric, ceiling=ceiling, train_size=train_size)
        score = score.expand_dims('train_size')
        score['train_size'] = [train_size]
        ceilings.append(score)
    ceilings = merge_data_arrays(ceilings)

    # plot
    def sigmoid(x, a, b, c):
        return 1 / (1 + np.exp(-b * (x - a))) + c

    fig, axes = pyplot.subplots(1, 2, figsize=(10, 5))
    for ax, region in zip(axes, np.unique(ceilings['region'])):
        ax.set_title(region)
        x = np.array(train_sizes)
        score = ceilings.sel(region=region)
        y = score.sel(aggregation='center').values
        err = score.sel(aggregation='error').values
        ax.scatter(x, y)
        ax.errorbar(x, y, err, linestyle='None')

        fit_params, fit_cov = scipy.optimize.curve_fit(sigmoid, x, y)
        fit_y = sigmoid(x, *fit_params)
        r, p = scipy.stats.pearsonr(y, fit_y)
        assert p < 0.05
        x_fitplot = np.arange(min(x), max(x), (max(x) - min(x)) / 100)
        # x_fitplot = np.arange(0, 100)  # extrapolate
        ax.plot(x_fitplot, sigmoid(x_fitplot, *fit_params))
    ax = fig.add_subplot(111, frameon=False)
    pyplot.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel('train size')
    ax.set_ylabel('goodness-of-fit')
    pyplot.tight_layout()
    return fig


def compute_ceilings(assembly_name, metric_name, ceiling, train_size):
    assembly, metric = instantiate_benchmark(assembly_name, metric_name)

    scores = []
    dividers = np.unique(assembly['region'])
    for i, region in enumerate(dividers):
        _logger.debug("dividers {}/{}: region={}".format(i + 1, len(dividers), region))
        score = compute_ceiling(assembly_name, metric_name, ceiling=ceiling, train_size=train_size, region=region)
        score = score.aggregation
        score = score.expand_dims('region')
        score['region'] = [region]
        scores.append(score)
    scores = merge_data_arrays(scores)
    return scores


@store()
def compute_ceiling(assembly, metric, ceiling, train_size, region):
    _assembly, _metric = instantiate_benchmark(assembly, metric)
    _assembly = _assembly.multisel(region=region)
    ceiling = ceilings[ceiling](metric, repetition_train_size=train_size)
    score = ceiling(_assembly)
    return score


@cache()
def instantiate_benchmark(data, metric):
    benchmark = load_neural_benchmark(data, metric)
    data = benchmark._target_assembly
    metric = benchmark._metric
    return data, metric


if __name__ == '__main__':
    for ceiling in ['cons', 'splitrep']:
        plot(ceiling=ceiling)
        pyplot.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results',
                                    'ceiling.train_size-{}.png'.format(ceiling)))
