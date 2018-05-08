import logging
import sys

import numpy as np
import seaborn
from matplotlib import pyplot

import neurality
from neurality import score_physiology, model_layers
from neurality.plot import shaded_errorbar

seaborn.set()


def plot(model, layers=None, neural_data=neurality.Defaults.neural_data, ax=None):
    if ax is None:
        fig, ax = pyplot.subplots()
    layers = layers or model_layers[model]
    score = score_physiology(model=model, layers=layers, neural_data=neural_data)
    regions = np.unique(score.center['region'])
    for region in regions:
        region_data = score.center.sel(region=region)
        errs = score.error.sel(region=region)
        x, y, error = region_data['layer'].values, region_data.values, errs.values
        shaded_errorbar(x, y, error, ax=ax)
        ax.set_xticklabels(x, rotation=90)
    pyplot.legend(regions)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plot('alexnet')
    pyplot.savefig('results/layerfit-alexnet.jpg')
