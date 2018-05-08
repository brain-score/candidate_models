import logging
import sys

import seaborn

import neurality
from neurality import score_physiology, model_layers
from neurality.plot import shaded_errorbar, region_color_mapping

from collections import defaultdict

import numpy as np

from collections import OrderedDict
from matplotlib import pyplot, patheffects

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


def plot_average_layer_ordering(models, neural_data=neurality.Defaults.neural_data, ax=None):
    region_model_rellayer_scores = defaultdict(dict)

    regions = None
    # save relative scores of each layer in each model
    for model in models:
        print("Model {}".format(model))
        score = score_physiology(model=model, layers=model_layers[model], neural_data=neural_data)
        regions = score.center['region'].values
        for region in regions:
            print("Region {}".format(region))
            region_model_rellayer_scores[region][model] = {}
            layers = score.center['layer'].values
            for i, layer in enumerate(layers):
                relative_position = i / len(layers)
                center = score.center.sel(region=region).sel(layer=layer).values
                error = score.error.sel(region=region).sel(layer=layer).values
                region_model_rellayer_scores[region][model][relative_position] = center, error

    if ax is None:
        fig, ax = pyplot.subplots()
    # compute trend for each region and each model separately and then average trends for all models
    for region in regions:
        model_relative_scores = region_model_rellayer_scores[region]
        fits = []
        for model, relative_scores in model_relative_scores.items():
            relative_scores = OrderedDict((relative, score) for relative, score in
                                          sorted(zip(relative_scores.keys(), relative_scores.values()),
                                                 key=lambda kv: kv[0]))
            x = list(relative_scores.keys())
            y = np.array([mean for mean, std in relative_scores.values()])
            ax.scatter(x, y, color=region_color_mapping[region], alpha=1 / len(x))
            # fit trend to model curve
            z = np.polyfit(x, y, 2)
            fits.append(z)
            f = np.poly1d(z)
            y_fit = f(x)
            ax.plot(x, y_fit, color=region_color_mapping[region], linewidth=2., alpha=0.3, label=region)
        # plot average of all fits
        z = np.mean(np.array(fits), axis=0)
        f = np.poly1d(z)
        y_fit = f(x)
        ax.plot(x, y_fit, color=region_color_mapping[region], linewidth=4., label=region)

    txts = [pyplot.text(i * 0.1, 0.55, region, fontsize=16, weight='bold', color=region_color_mapping[region])
            for i, region in enumerate(regions)]
    for txt in txts:
        txt.set_path_effects([patheffects.withStroke(linewidth=5, foreground='w')])
    pyplot.xlabel('Relative layer position in models', fontsize=16, style='italic')
    pyplot.ylabel('Neural fit', fontsize=16, style='italic')
    pyplot.tight_layout()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plot('alexnet')
    pyplot.savefig('results/layerfit-alexnet.jpg')
