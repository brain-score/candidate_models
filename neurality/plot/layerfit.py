import logging
import sys

import numpy as np
import seaborn
from matplotlib import pyplot

import neurality
from neurality import score_physiology, model_layers

seaborn.set()


def plot(model, layers=None, neural_data=neurality.Defaults.neural_data):
    layers = layers or model_layers[model]
    score = score_physiology(model=model, layers=layers, neural_data=neural_data)
    regions = np.unique(score.center['region'])
    for region in regions:
        region_data = score.center.sel(region=region)
        x = region_data['layer'].values
        errs = score.error.sel(region=region)
        pyplot.errorbar(x=x, y=region_data.values, yerr=errs.values)
        ax = pyplot.gca()
        ax.set_xticklabels(x, rotation=90)
    pyplot.legend(regions)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plot('alexnet')
    pyplot.savefig('results/layerfit-alexnet.jpg')
