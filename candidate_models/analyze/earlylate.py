import itertools
import logging
import os
import sys

import pandas as pd
import seaborn

from candidate_models import score_model
from candidate_models.analyze.coco import collect_old_scores

seaborn.set()
from matplotlib import pyplot

from candidate_models.analyze import align, DataCollector


def layerwise_scores(model_layer, benchmark):
    def get_model_score(model, layer=None):
        score = score_model(model, benchmark=benchmark)
        score = score.sel(layer=layer)
        if isinstance(layer, str):
            score = score.expand_dims('layer')
            score['layer'] = [layer]
        result = []
        for layer, region, time_bin_start in itertools.product(
                score['layer'].values, score['region'].values, score['time_bin_start'].values):
            _score = score.sel(layer=layer, region=region, time_bin_start=time_bin_start)
            result.append({'model': model, 'layer': layer,
                           'region': region, 'time_bin_start': time_bin_start,
                           'score': _score.sel(aggregation='center').values,
                           'error': _score.sel(aggregation='error').values})
        return result

    earlylate_scores = [get_model_score(model, layer) for model, layer in model_layer.items()]
    earlylate_scores = pd.DataFrame(list(itertools.chain(*earlylate_scores)))
    return earlylate_scores


def ko_ost_plot(models=None,
                benchmark='dicarlo.Majaj2015.earlylate-ost',
                early=(90, 110), late=(190, 210)):
    models = models or {
        # 'alexnet': 'features.12',
        # 'vgg-16': 'block4_pool',
        'resnet-50_v2': 'resnet_v2_50/block4/unit_1/bottleneck_v2',
        'resnet-101_v2': 'global_pool',
        # 'cornet_r2': [f'block4.relu3-t{timestep}' for timestep in range(5)],
        # 'cornet_s': ['V4.output-t0', 'V4.output-t1', 'V4.output-t2', 'V4.output-t3'],
        'cornet_s': ['IT.output-t0', 'IT.output-t1'],
    }

    earlylate_scores = layerwise_scores(models, benchmark)
    earlylate_scores = earlylate_scores[earlylate_scores['region'] == 'IT']

    fig, axes = pyplot.subplots(ncols=2)

    def plot(ax, time_bin):
        data = earlylate_scores[earlylate_scores['time_bin_start'] == time_bin[0]]
        identifiers = [f"{row.model}/{row.layer[-15:]}" for row in data.itertuples()]
        x = list(range(len(identifiers)))
        bottom = .3
        ax.bar(x=x, height=data['score'] - bottom, bottom=bottom)
        ax.errorbar(x=x, y=data['score'], yerr=data['error'], fmt=' ')
        ax.set_xticks(x)
        ax.set_xticklabels(identifiers, rotation=90)
        ax.set_title(f"IT {time_bin}")

    plot(axes[0], early)
    plot(axes[1], late)

    pyplot.tight_layout()
    fig.subplots_adjust(bottom=.5)
    # pyplot.savefig(f'results/earlylate-ko-{early}-{late}.png')
    pyplot.savefig(f'results/earlylate-ko-ost-{early}-{late}.png')


def ko_plot(models=None,
            benchmark='dicarlo.Majaj2015.earlylate',
            region='IT',
            early=(90, 110), late=(190, 210)):
    model_defaults = {'alexnet': 'features.12',
                      'vgg-16': 'block4_pool',
                      'resnet-50_v2': 'resnet_v2_50/block4/unit_1/bottleneck_v2',
                      'resnet-101_v2': 'global_pool',
                      'cornet_r2': [f'block{4 if region == "IT" else 3}.relu3-t{timestep}' for timestep in range(5)],
                      'cornet_s': ['IT.output-t0', 'IT.output-t1'] if region == 'IT' else \
                          ['V4.output-t0', 'V4.output-t1', 'V4.output-t2', 'V4.output-t3']}
    models = models or model_defaults

    earlylate_scores = layerwise_scores(models, benchmark)
    earlylate_scores = earlylate_scores[earlylate_scores['region'] == region]

    fig, axes = pyplot.subplots(ncols=2)

    def plot(ax, time_bin):
        data = earlylate_scores[earlylate_scores['time_bin_start'] == time_bin[0]]
        identifiers = [f"{row.model}/{row.layer[-15:]}" for row in data.itertuples()]
        x = list(range(len(identifiers)))
        bottom = .3 if region == 'IT' else .15
        ax.bar(x=x, height=data['score'] - bottom, bottom=bottom)
        ax.errorbar(x=x, y=data['score'], yerr=data['error'], fmt=' ')
        ax.set_xticks(x)
        ax.set_xticklabels(identifiers, rotation=90)
        ax.set_title(f"{region} {time_bin}")

    plot(axes[0], early)
    plot(axes[1], late)

    pyplot.tight_layout()
    fig.subplots_adjust(bottom=.5)
    pyplot.savefig(f'results/earlylate-ko-{region}-{early}-{late}.png')


def layerfixed_scores(models, benchmark):
    old_scores = collect_old_scores(os.path.join(os.path.dirname(__file__), '..', '..',
                                                 'output', 'candidate_models._score_physiology'))
    old_scores = old_scores[old_scores['benchmark'] == 'dicarlo.Majaj2015.IT']

    def get_model_score(model):
        score = score_model(model, benchmark=benchmark)
        hvm_layer = old_scores[old_scores['model'] == model]['layer']
        layer_score = score.sel(layer=hvm_layer.iloc[0])
        result = []
        for region, time_bin_start in itertools.product(
                layer_score['region'].values, layer_score['time_bin_start'].values):
            _score = layer_score.sel(region=region, time_bin_start=time_bin_start)
            result.append({'model': model, 'region': region, 'time_bin_start': time_bin_start,
                           'score': _score.sel(aggregation='center').values,
                           'error': _score.sel(aggregation='error').values})
        return result

    earlylate_scores = list(itertools.chain(*[get_model_score(model) for model in models]))
    earlylate_scores = pd.DataFrame(earlylate_scores)
    return earlylate_scores


def plot_same_layers(models, benchmark='dicarlo.Majaj2015.earlylate', early=(90, 110), late=(190, 210)):
    classification_scores = DataCollector()()
    classification_scores = classification_scores[classification_scores['benchmark'] == 'ImageNet']
    earlylate_scores = layerfixed_scores(models, benchmark)

    fig, axes = pyplot.subplots(ncols=2, nrows=2)
    for ax, (region, time_bin_start) in zip(axes, itertools.product(['V4', 'IT'], [early[0], late[0]])):
        neural_scores = earlylate_scores[(earlylate_scores['region'] == region) &
                                         (earlylate_scores['time_bin_start'] == time_bin_start)]
        neural_scores = align(neural_scores, classification_scores, on='model')
        x, xerr = classification_scores['score'], classification_scores['error']
        y, yerr = neural_scores['score'], neural_scores['error']
        colors = ['b' if not model.startswith('cornet') else 'r' for model in neural_scores['model']]
        ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, linestyle=' ', marker='.', ecolor=colors)
        ax.set_xlabel(f'early {early}')
        ax.set_ylabel(f'late {late}')
        if region == 'IT':
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
        ax.set_title(region)
    pyplot.tight_layout()
    pyplot.savefig(f'results/earlylate-imagenet-{early}-{late}.png')


def plot(data, early, late):
    early_scores = data[data['time_bin_start'] == early[0]]
    late_scores = data[data['time_bin_start'] == late[0]]
    fig, axes = pyplot.subplots(ncols=2)
    for region, ax in zip(['V4', 'IT'], axes):
        region_early_scores = early_scores[data['region'] == region]
        region_late_scores = late_scores[data['region'] == region]
        region_late_scores = align(region_late_scores, region_early_scores, on='model')
        x, xerr = region_early_scores['score'], region_early_scores['error']
        y, yerr = region_late_scores['score'], region_late_scores['error']
        colors = ['b' if not model.startswith('cornet') else 'r' for model in region_early_scores['model']]
        ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, linestyle=' ', marker='.', ecolor=colors)
        ax.set_xlabel(f'early {early}')
        ax.set_ylabel(f'late {late}')
        if region == 'IT':
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
        ax.set_title(region)
    pyplot.tight_layout()
    pyplot.savefig(f'results/earlylate-{early}-{late}.png')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # ko_plot()
    # ko_plot(region='V4')
    # ko_ost_plot()

    data = DataCollector()()

    plot_same_layers(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate']['model'])
    # plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate'], early=(90, 110), late=(190, 210))
    # plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate-alternatives'], early=(70, 90), late=(170, 190))
    # plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate-alternatives'], early=(70, 90), late=(150, 250))
