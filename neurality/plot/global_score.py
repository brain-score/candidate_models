import itertools
import logging
import os
import sys
from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter

import neurality
from mkgu.assemblies import merge_data_arrays
from mkgu.metrics import MeanScore
from neurality import score_physiology, model_layers
from neurality.plot import score_color_mapping, get_models, clean_axis

model_meta = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'models', 'models.csv'))
model_performance = {row['model']: row['top1'] for _, row in model_meta.iterrows()}
model_performance['alexnet'] = 1 - 0.4233
model_performance['squeezenet'] = 0.575
model_depth = {row['model']: row['depth'] for _, row in model_meta.iterrows()}
model_nonlins = {row['model']: row['nonlins'] for _, row in model_meta.iterrows()}
model_flops = {row['model']: row['flops'] for _, row in model_meta.iterrows()}
model_params = {row['model']: row['num_params'] for _, row in model_meta.iterrows()}
model_behavior = {row['model']: row['behav_r'] for _, row in model_meta.iterrows()}
model_behavior['squeezenet'] = model_behavior['alexnet']  # FIXME: remove mock data

_model_years = {
    'alexnet': 2012,
    'vgg': 2014,
    'resnet': 2015,
    'inception_v': 2015,
    'mobilenet': 2017,
    'densenet': 2016,
    'squeezenet': 2016,
    'inception_resnet_v2': 2016,
    'nasnet': 2017,
}

ceilings = {
    'V4 RDM': .941,
    'V4 neural': .892,
    'IT RDM': .895,
    'IT neural': .817,
    'behavioral': .479,
}

model_years = {}
for model in model_performance:
    year = [year for prefix, year in _model_years.items() if model.startswith(prefix)]
    if len(year) == 0:
        continue
    model_years[model] = year[0]


class Mode(Enum):
    YEAR_VS_SCORE = 1
    PERFORMANCE_VS_SCORE = 2
    DEPTH_VS_SCORE = 3
    NONLINS_VS_SCORE = 4
    FLOPS_VS_SCORE = 5
    PARAMS_VS_SCORE = 6


def parse_neural_data(models, neural_data=neurality.Defaults.neural_data):
    savepath = os.path.join(os.path.dirname(__file__), 'neural.csv')
    if os.path.isfile(savepath):
        return pd.read_csv(savepath)

    regions = ['V4', 'IT']
    metrics = ['neural_fit']  # , 'rdm']
    data = {'model': models}
    for region, metric in itertools.product(regions, metrics):
        neural_centers, neural_errors = [], []
        for model in models:
            layers = model_layers[model] if not model.startswith('basenet') \
                else ['basenet-layer']  # ['basenet-layer_v4', 'basenet-layer_it']
            neural_score = score_physiology(model=model, layers=layers, neural_data=neural_data, metric_name=metric)
            neural_score = MeanScore(neural_score.values)  # re-package with SEM
            center, error = neural_score.center.max(dim='layer'), neural_score.error.max(dim='layer')
            center, error = center.expand_dims('model'), error.expand_dims('model')
            center['model'], error['model'] = [model], [model]
            neural_centers.append(center)
            neural_errors.append(error)
        neural_centers, neural_errors = merge_data_arrays(neural_centers), merge_data_arrays(neural_errors)
        data['neural: {}-{}'.format(region, metric)] = neural_centers.sel(region=region).values
        data['neural-error: {}-{}'.format(region, metric)] = neural_errors.sel(region=region).values

    data = pd.DataFrame(data=data)
    data['neural-mean'] = (data['neural: V4-neural_fit'] + data['neural: IT-neural_fit']) / 2
    data.to_csv(savepath)
    return data


def prepare_data(models, neural_data):
    data = parse_neural_data(models, neural_data)
    data['i2n'] = [model_behavior[model] if model in model_behavior else np.nan for model in data['model']]
    data['performance'] = [100 * model_performance[model] for model in data['model']]

    global_scores = [[data['neural-mean'][data['model'] == model].values[0],
                      data['i2n'][data['model'] == model].values[0]]
                     for model in data['model']]
    data['global'] = np.mean(global_scores, axis=1)
    data = data[data['model'].isin(models)]
    # rank - yields non-integers when there are multiple winners
    data['rank:neural:V4'] = data['neural: V4-neural_fit'].rank(ascending=False)
    data['rank:neural:IT'] = data['neural: IT-neural_fit'].rank(ascending=False)
    data['rank:behavior'] = data['i2n'].rank(ascending=False)
    data['rank:neural'] = (data['rank:neural:V4'] + data['rank:neural:IT']) / 2
    data['rank:neural'] = data['rank:neural'].rank(ascending=True)
    data['rank'] = (data['rank:neural'] + data['rank:behavior']) / 2
    data['rank'] = data['rank'].rank(ascending=True)

    # latex table
    table = data[[not model.startswith('basenet') for model in data['model']]]
    table = table[['rank', 'model', 'neural: V4-neural_fit', 'neural: IT-neural_fit', 'i2n', 'performance']]
    table = table.sort_values('rank')
    table = table.apply(highlight_max)
    table.to_latex('data.tex', escape=False)
    return data


def highlight_max(data):
    assert data.ndim == 1
    is_max = data == data.max()
    if isinstance(next(iter(data)), str):
        return [x.replace('_', '\\_') if isinstance(x, str) else x for x in data]
    all_ints = all(x.is_integer() for x in data)
    data = ["{:.02f}".format(x) if not all_ints else "{:.0f}".format(x) for x in data]  # format comma
    return [('\\textbf{' + str(x) + '}') if _is_max else x for x, _is_max in zip(data, is_max)]


def basenet_filter(data):
    return [row['model'].startswith('basenet') for _, row in data.iterrows()]


def plot_all(models, neural_data=neurality.Defaults.neural_data, mode=Mode.YEAR_VS_SCORE,
             fit_method='fit', logx=False):
    data = prepare_data(models, neural_data)

    if mode == Mode.YEAR_VS_SCORE:
        x = [model_years[model] for model in models]
        xlabel = 'Year'
    elif mode == Mode.PERFORMANCE_VS_SCORE:
        x = data['performance']
        xlabel = 'Imagenet performance (% top-1)'
    elif mode == Mode.DEPTH_VS_SCORE:
        x = [model_depth[model] for model in models]
        xlabel = 'Model depth'
    elif mode == Mode.NONLINS_VS_SCORE:
        x = [model_nonlins[model] for model in models]
        xlabel = 'Model non-linearities'
    elif mode == Mode.FLOPS_VS_SCORE:
        x = [model_flops[model] for model in models]
        xlabel = 'FLOPs'
    elif mode == Mode.PARAMS_VS_SCORE:
        x = [model_params[model] for model in models]
        xlabel = 'Model params'
    else:
        raise ValueError("invalid mode {}".format(mode))

    data['x'] = x
    basenets = basenet_filter(data)
    nonbasenets = [not basenet for basenet in basenets]
    basenet_alpha = 0.3
    nonbasenet_alpha = 0.7

    figs = {}

    def xtick_formatter(x, pos, pad=True):
        """Format 1 as 1, 0 as 0, and all values whose absolute values is between
        0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
        formatted as -.4)."""
        val_str = '{:.02f}'.format(x) if pad else '{:.2g}'.format(x)
        if 0 <= np.abs(x) <= 1:
            return val_str.replace("0", "", 1)
        else:
            return val_str

    major_formatter = FuncFormatter(xtick_formatter)

    def post(ax, formatter=major_formatter):
        # ax.set_ylim([0, .85])
        ax.set_xlabel(xlabel)
        if formatter is not None:
            ax.yaxis.set_major_formatter(formatter)
        clean_axis(ax)
        pyplot.tight_layout()
        if logx:
            ax.set_xscale('log')

    # neural
    regions = ['V4', 'IT']
    metrics = ['neural_fit']  # , 'rdm']
    for region, metric in itertools.product(regions, metrics):
        fig, ax = pyplot.subplots()
        plot_neural(ax=ax, data=data[basenets], fit_method=fit_method, region=region, metric=metric,
                    color=score_color_mapping['basenet'], scatter_alpha=basenet_alpha)
        plot_neural(ax=ax, data=data[nonbasenets], fit_method=fit_method, region=region, metric=metric,
                    scatter_alpha=nonbasenet_alpha)
        ax.set_ylabel('neural: ' + metric)
        post(ax)
        figs['neural-' + region + '-' + metric] = fig

    # behavior
    for metric in ['i2n']:
        fig, ax = pyplot.subplots()
        plot_behavior(ax, data[basenets], fit_method, metric=metric,
                      color=score_color_mapping['basenet'], scatter_alpha=basenet_alpha)
        plot_behavior(ax, data[nonbasenets], fit_method, metric=metric, scatter_alpha=nonbasenet_alpha)
        ax.set_ylabel('behavior: ' + metric)
        post(ax)
        figs['behavior-' + metric] = fig

    # global
    fig, ax = pyplot.subplots()
    plot_global(ax, data[basenets], fit_method, color=score_color_mapping['basenet'], scatter_alpha=basenet_alpha)
    plot_global(ax, data[nonbasenets], fit_method, scatter_alpha=nonbasenet_alpha)
    # basenets: linear fit
    x, y = data[basenets]['x'], data[basenets]['rank']
    indices = np.argsort(x)
    x, y = np.array(x)[indices], np.array(y)[indices]
    z = np.polyfit(x, y, 1)
    f = np.poly1d(z)
    x = np.append(x, 80)
    y_fit = f(x)
    ax.plot(x, y_fit, color=score_color_mapping['basenet'], linewidth=2., linestyle='dashed')
    # SOTA: curve fit
    x, y = data[nonbasenets]['x'], data[nonbasenets]['rank']
    indices = np.argsort(x)
    x, y = np.array(x)[indices], np.array(y)[indices]
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)
    # for i in reversed(range(int(min(x)))):
    #     x = np.insert(x, 0, i)
    y_fit = f(x)
    ax.plot(x, y_fit, color=score_color_mapping['global'], linewidth=2.)

    post(ax, formatter=None)  # major_formatter_nopad)
    ax.invert_yaxis()
    ax.set_ylabel('Rank')
    ax.set_ylim(ax.get_ylim()[:1] + (-150,))
    ax.set_yticklabels(['' if x < 0 else '{:.0f}'.format(x) if x != 0 else '1' for x in ax.get_yticks()])
    figs['global'] = fig

    # global-zoom
    zoom = data[[row['performance'] > 70 for _, row in data.iterrows()]]
    fig, ax = pyplot.subplots()
    plot_global(ax, zoom, fit_method, marker_size=75)
    post(ax)
    figs['global-zoom'] = fig

    # performance
    if mode == Mode.YEAR_VS_SCORE:
        y = data['performance'].values
        fig, ax = pyplot.subplots()
        line = _plot_score(x, y, label='performance', fit_method=fit_method, ax=ax)
        ax.set_xlabel(xlabel)
        figs.append(fig)

    pyplot.tight_layout()
    return figs


def plot_global(ax, data, fit_method, **kwargs):
    x = data['x']
    y = data['rank'].values
    return _plot_score(x, y, label='global', fit_method=fit_method, label_long='BrainRank', ax=ax, **kwargs)


def plot_behavior(ax, data, fit_method, metric, **kwargs):
    x = data['x']
    y = data[metric]
    return _plot_score(x, y, label='behavior', ax=ax, fit_method=fit_method, fit_kwargs=dict(linestyle='dashed'),
                       **kwargs)


def plot_neural(ax, data, fit_method, region, metric, **kwargs):
    x = data['x']
    y, error = data['neural: {}-{}'.format(region, metric)], data['neural-error: {}-{}'.format(region, metric)]
    return _plot_score(x, y, error=error, label=region, label_long='neural: {}-{}'.format(region, metric), ax=ax,
                       fit_method=fit_method, fit_kwargs=dict(linestyle='dashed'), **kwargs)


def _plot_score(x, y, label, ax, error=None, label_long=None, fit_method='fit',
                fit_kwargs=None, color=None, marker_size=20, scatter_alpha=0.3):
    label_long = label_long or label
    fit_kwargs = fit_kwargs or {}
    color = color or score_color_mapping[label]
    ax.scatter(x, y, label=label_long, color=color, alpha=scatter_alpha, s=marker_size)
    ax.errorbar(x, y, error, label=label_long, color=color, alpha=scatter_alpha,
                elinewidth=1, linestyle='None')
    # fit trend
    if fit_method:
        indices = np.argsort(x)
        x, y = np.array(x)[indices], np.array(y)[indices]
        if fit_method == 'fit':
            z = np.polyfit(x, y, 2)
            f = np.poly1d(z)
            y_fit = f(x)
        elif fit_method == 'convex':
            y_fit = []
            curr_x, curr_y = 0, -99999
            for _x, _y in zip(x, y):
                if _x != curr_x:
                    curr_x = _x
                    curr_y = _y
                elif _y > curr_y:
                    curr_y = _y
                y_fit.append(curr_y)
        line = ax.plot(x, y_fit, color=color, linewidth=2., label=label_long, **fit_kwargs)
        return line[0]


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fit = None

    models = get_models()
    missing_models = set(model_behavior.keys()) - set(models)
    print("Missing models:", missing_models)

    runs = {'performance': Mode.PERFORMANCE_VS_SCORE, 'depth': Mode.DEPTH_VS_SCORE, 'nonlins': Mode.NONLINS_VS_SCORE,
            'flops': Mode.FLOPS_VS_SCORE, 'numparams': Mode.PARAMS_VS_SCORE}
    for label, mode in runs.items():
        figs = plot_all(models, mode=mode, fit_method=fit, logx=label not in ['performance'])
        for name, fig in figs.items():
            savepath = 'results/scores/{}-{}.pdf'.format(label, name)
            fig.savefig(savepath)
            print("Saved to", savepath)
        break
