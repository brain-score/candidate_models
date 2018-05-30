import logging
import os
import sys
from collections import OrderedDict
from collections import defaultdict

import numpy as np
from matplotlib import pyplot

import neurality
from neurality import score_physiology, model_layers
from neurality.plot import shaded_errorbar, score_color_mapping, get_models, clean_axis


def plot(model, layers=None, neural_data=neurality.Defaults.neural_data, ax=None):
    if ax is None:
        fig, ax = pyplot.subplots()
    layers = layers or model_layers[model]
    score = score_physiology(model=model, layers=layers, neural_data=neural_data)
    regions = ['V4', 'IT']
    for region in regions:
        region_data = score.center.sel(region=region)
        errs = score.error.sel(region=region)
        x, y, error = region_data['layer'].values, region_data.values, errs.values
        shaded_errorbar(x, y, error, ax=ax, color=score_color_mapping[region])
        if model == 'alexnet':
            x = ['conv{}'.format(i + 1) for i in range(5)] + ['fc{}'.format(i + 1) for i in range(5, 7)]
        ax.set_xticklabels(x, rotation=90)
    pyplot.legend(regions, ncol=2, loc='lower center')
    ax.set_ylabel('Neural Fit')
    clean_axis(ax)
    pyplot.tight_layout()


def plot_average_layer_ordering(models, neural_data=neurality.Defaults.neural_data, ax=None):
    region_model_rellayer_scores = defaultdict(dict)

    # save relative scores of each layer in each model
    for model in models:
        print("Model {}".format(model))
        layers = model_layers[model] if not model.startswith('basenet') else ['basenet-layer_v4', 'basenet-layer_pit', 'basenet-layer_ait']
        score = score_physiology(model=model, layers=layers, neural_data=neural_data)
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
    lines = []
    regions = ['V4', 'IT']
    for region in regions:
        model_relative_scores = region_model_rellayer_scores[region]
        fits = []
        for model, relative_scores in model_relative_scores.items():
            relative_scores = OrderedDict((relative, score) for relative, score in
                                          sorted(zip(relative_scores.keys(), relative_scores.values()),
                                                 key=lambda kv: kv[0]))
            x = list(relative_scores.keys())
            y = np.array([mean for mean, std in relative_scores.values()])
            # ax.scatter(x, y, color=score_color_mapping[region], alpha=1 / len(x))
            # fit trend to model curve
            z = np.polyfit(x, y, 2)
            fits.append(z)
            f = np.poly1d(z)
            y_fit = f(x)
            ax.plot(x, y_fit, color=score_color_mapping[region], linewidth=1., alpha=.1, label=region)
        # plot average of all fits
        z = np.mean(np.array(fits), axis=0)
        f = np.poly1d(z)
        y_fit = f(x)
        line = ax.plot(x, y_fit, color=score_color_mapping[region], linewidth=3., label=region)
        lines.append(line[0])

    pyplot.legend(lines, regions, loc='lower center', ncol=len(lines))
    pyplot.xlabel('Relative layer position in models', fontsize=16, style='italic')
    pyplot.ylabel('Neural fit', fontsize=16, style='italic')
    clean_axis(ax)
    pyplot.tight_layout()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    models = get_models()
    models = [model for model in models if not model.startswith('basenet')]
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'layerfit')
    plot('alexnet')
    pyplot.savefig(os.path.join(results_dir, 'alexnet.pdf'))
    plot_average_layer_ordering(models)
    pyplot.savefig(os.path.join(results_dir, 'ordering.pdf'))
