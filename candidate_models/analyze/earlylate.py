import itertools
import logging
import sys

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot

from candidate_models import score_model, model_layers
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
    mean70170_scores = DataCollector()()
    mean70170_scores = mean70170_scores[(mean70170_scores['benchmark'] == 'dicarlo.Majaj2015.V4') |
                                        (mean70170_scores['benchmark'] == 'dicarlo.Majaj2015.IT')]

    def get_model_score(model):
        score = score_model(model, benchmark=benchmark)
        result = []
        for region, time_bin_start in itertools.product(
                score['region'].values, score['time_bin_start'].values):
            mean_region_scores = mean70170_scores[mean70170_scores['benchmark'] == f'dicarlo.Majaj2015.{region}']
            selection_score = score.sel(region=region, time_bin_start=time_bin_start)
            if not model.startswith('cornet'):
                layer = mean_region_scores[mean_region_scores['model'] == model]['layer']
                assert len(layer) == 1
                layer = layer.iloc[0]
                layer_score = selection_score.sel(layer=layer)
            else:  # choose IT/V4 for CORnet
                layers = [layer for layer in model_layers[model] if layer.startswith(region)]
                layer_score = selection_score.sel(layer=layers)
                argmax = layer_score.sel(aggregation='center').argmax('layer')
                layer_score = layer_score[{'layer': argmax.values}]

            result.append({'model': model, 'region': region, 'time_bin_start': time_bin_start,
                           'score': layer_score.sel(aggregation='center').values,
                           'error': layer_score.sel(aggregation='error').values})
        return result

    earlylate_scores = list(itertools.chain(*[get_model_score(model) for model in models]))
    earlylate_scores = pd.DataFrame(earlylate_scores)
    return earlylate_scores


def plot_same_layers(models=None, benchmark='dicarlo.Majaj2015.earlylate', early=(90, 110), late=(190, 210),
                     save=False):
    if not models:
        data = DataCollector()()
        models = [model for model in data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate']['model']
                  if model not in ['cornet_z', 'cornet_r', 'cornet_r2']]  # no Imagenet scores
        # also, for cornet_r2, avoid dealing with layers that are not named after regions
        models = set(models)

    classification_scores = DataCollector()()
    classification_scores = classification_scores[classification_scores['benchmark'] == 'ImageNet']
    classification_scores = classification_scores[classification_scores['model'].isin(models)]
    earlylate_scores = layerfixed_scores(models, benchmark)

    fig, axes = pyplot.subplots(ncols=2, nrows=2)
    for ax, (region, time_bin_start) in zip(axes.flatten(), itertools.product(['V4', 'IT'], [early[0], late[0]])):
        neural_scores = earlylate_scores[(earlylate_scores['region'] == region) &
                                         (earlylate_scores['time_bin_start'] == time_bin_start)]
        for with_cornet in [False, True]:
            current_classification_scores = classification_scores[
                [model.startswith('cornet') == with_cornet for model in classification_scores['model']]]
            current_neural_scores = align(neural_scores, current_classification_scores, on='model')
            x, xerr = current_classification_scores['score'], current_classification_scores['error']
            y, yerr = current_neural_scores['score'], current_neural_scores['error']
            assert not np.any([np.isnan(val.astype(np.float)).any() for val in [x, xerr, y, yerr]])
            color = '#808080' if not with_cornet else '#D4145A'
            ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, linestyle=' ', marker='.', color=color, ecolor=color)
        ax.set_xlabel('ImageNet top-1')
        is_early = time_bin_start == early[0]
        # ax.set_ylabel(f"early {early}" if is_early else f"late {late}")
        if not is_early:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
        seaborn.despine(ax=ax, right=is_early, left=not is_early, top=True)
        if region == 'V4':
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
        ax.set_title(f"{region} {'early' if is_early else 'late'} {early if is_early else late}")
    if save:
        pyplot.tight_layout()
        pyplot.savefig(f'results/earlylate-imagenet-{early}-{late}.png')
    return fig


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

    plot_same_layers(save=True)
    # plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate'], early=(90, 110), late=(190, 210))
    # plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate-alternatives'], early=(70, 90), late=(170, 190))
    # plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate-alternatives'], early=(70, 90), late=(150, 250))
