import logging
import sys

import seaborn

import neurality
from mkgu.assemblies import merge_data_arrays
from neurality import score_physiology, model_layers
from neurality.plot import shaded_errorbar, region_color_mapping

from collections import defaultdict

import numpy as np

from collections import OrderedDict
from matplotlib import pyplot, patheffects

seaborn.set()

model_years = {
    'alexnet': 2012,
    'vgg16': 2014,
    'vgg19': 2014,
    'resnet50': 2015,
    'resnet152': 2015,
    'inception_v3': 2015,
    'mobilenet': 2017,
    'densenet': 2016,
    'squeezenet': 2016,
    'inception_resnet_v2': 2016,
}

model_top1imagenet = {
    'alexnet': 37.5,
    'vgg16': 24.4,
    'resnet50': 22.85,
    'inception_v3': 18.77,
    'densenet': 23.61,
    'vgg19': -1,
    'resnet152': -1,
    'mobilenet': -1,
    'squeezenet': -1,
    'inception_resnet_v2': -1,
}


def plot(models, neural_data=neurality.Defaults.neural_data, ax=None):
    if ax is None:
        fig, ax = pyplot.subplots()

    score_centers, score_errors = [], []
    for model in models:
        layers = model_layers[model]
        score = score_physiology(model=model, layers=layers, neural_data=neural_data)
        center, error = score.center.max(dim='layer'), score.error.max(dim='layer')
        center, error = center.expand_dims('model'), error.expand_dims('model')
        center['model'], error['model'] = [model], [model]
        score_centers.append(center)
        score_errors.append(error)
    score_centers, score_errors = merge_data_arrays(score_centers), merge_data_arrays(score_errors)

    regions = np.unique(score.center['region'])
    plot_lines = []
    for region in regions:
        centers, errs = score_centers.sel(region=region), score_errors.sel(region=region)
        y, error = centers.values, errs.values
        x = [model_years[model] for model in centers['model'].values]
        line = ax.errorbar(x, y, error, label=region, fmt='o', color=region_color_mapping[region])
        plot_lines.append(line)
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90)
    ax.set_ylim([0.5, 1])

    ax_performance = ax.twinx()
    line = ax_performance.scatter([model_years[model] for model in models],
                                  [model_top1imagenet[model] for model in models],
                                  label='Imagenet top-1 val', color='red')
    ax_performance.invert_yaxis()
    plot_lines.append(line)
    pyplot.legend(plot_lines, [l.get_label() for l in plot_lines], loc=0)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plot(['alexnet', 'vgg16', 'resnet50', 'inception_v3', 'densenet'])
    pyplot.savefig('results/globalscore.png')
