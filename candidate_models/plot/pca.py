import logging
import sys

import matplotlib
import seaborn
from matplotlib import pyplot

from brainscore.assemblies import merge_data_arrays
from candidate_models import models, score_physiology
from candidate_models.plot import shaded_errorbar

seaborn.set()


def plot_model_pcas(model, layer, model_weights=models.Defaults.model_weights, stimulus_set='dicarlo.Majaj2015',
                    pcas=(25, 50, 100, 200, 500, 1000, 1500, 2000, 4000)):
    centers, errs = [], []
    for pca in pcas:
        score = score_physiology(model=model, weights=model_weights, layers=[layer], pca_components=pca,
                                 neural_data=stimulus_set)
        center, err = score.center, score.error
        center, err = center.expand_dims('pca'), err.expand_dims('pca')
        center['pca'] = [pca]
        err['pca'] = [pca]
        centers.append(center)
        errs.append(err)
    centers, errs = merge_data_arrays(centers), merge_data_arrays(errs)
    centers, errs = centers.sel(region='IT').squeeze('layer'), errs.sel(region='IT').squeeze('layer')
    x, y, error = centers['pca'], centers.values, errs.values

    fig, ax = pyplot.subplots()
    shaded_errorbar(x, y, error, ax=ax)
    ax.set_ylim([0, 1])
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    return fig


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plot_model_pcas('alexnet', 'features.12')
    pyplot.savefig('results/plot.jpg')
