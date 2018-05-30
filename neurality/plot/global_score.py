import functools
import itertools
import logging
import os
import sys
from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter

pyplot.rcParams['svg.fonttype'] = 'none'

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
    'densenet': 2016,  # aug
    'squeezenet': 2016,  # feb
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
ceilings = {
    'neural: V4-neural_fit': .892,
    'neural: IT-neural_fit': .817,
    'i2n': .479
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
                else ['basenet-layer_v4', 'basenet-layer_pit', 'basenet-layer_ait']
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
    # rank
    data['rank'] = data['global'].rank(ascending=False)

    # latex table
    table = data[[not model.startswith('basenet') for model in data['model']]]
    table = table[['global', 'model', 'performance', 'neural: V4-neural_fit', 'neural: IT-neural_fit', 'i2n']]
    table = table.sort_values('global', ascending=False)
    table = table.rename(columns={'global': 'Mean Score'})
    table = table.apply(highlight_max)
    table.to_latex('data.tex', escape=False, index=False)
    return data


def highlight_max(data):
    assert data.ndim == 1
    is_max = data == data.max()
    if isinstance(next(iter(data)), str):
        return [x.replace('_', '\\_') if isinstance(x, str) else x for x in data]
    all_ints = all(isinstance(x, int) or x.is_integer() for x in data)
    data = ["{:.0f}".format(x) if all_ints else "{:.2f}".format(x) if x > 1 else "{:.03f}".format(x).lstrip('0')
            for x in data]  # format comma
    return [('\\textbf{' + str(x) + '}') if _is_max else x for x, _is_max in zip(data, is_max)]


def basenet_filter(data):
    return [row['model'].startswith('basenet') for _, row in data.iterrows()]


def plot_all(models, neural_data=neurality.Defaults.neural_data, mode=Mode.YEAR_VS_SCORE,
             fit_method='fit', logx=False):
    data = prepare_data(models, neural_data)
    highlighted_models = [
        'nasnet_large',
        'pnasnet_large',  # good performance
        'resnet-152_v2', 'densenet-169',  # best overall
        # 'alexnet',
        'resnet-50_v2',  # good V4
        # 'mobilenet_v2_0.75_224',  # best mobilenet
        # 'basenet6_100_1--380000',  # best basenet
        # 'mobilenet_v1_1.0.224', 'inception_v2',  # good IT
        'inception_v4',  # good i2n
        'vgg-16',  # bad
    ]

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

    ## best models

    for best in ['global', 'neural: V4-neural_fit', 'neural: IT-neural_fit', 'i2n', 'performance']:
        # best_model = data.loc[data[best].idxmax()]
        best_models = data.nlargest(3, best)
        print("## {}".format(best))
        for i, (_, best_model) in enumerate(best_models.iterrows()):
            print("Top {} model: {}".format(i + 1, best),
                  best_model[['model', 'global', 'neural: V4-neural_fit', 'neural: IT-neural_fit', 'i2n', 'rank',
                              'performance']])
        model = best_models.iloc[0]['model']
        layers = model_layers[model] if not model.startswith('basenet') \
            else ['basenet-layer_v4', 'basenet-layer_pit', 'basenet-layer_ait']
        neural_score = score_physiology(model=model, layers=layers)
        max = neural_score.center['layer'][neural_score.center.argmax('layer')]
        print("V4:", max.sel(region='V4').values)
        print("IT:", max.sel(region='IT').values)
        print()

    figs = {}

    def xtick_formatter(x, pos, pad=True):
        """Format 1 as 1, 0 as 0, and all values whose absolute values is between
        0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
        formatted as -.4)."""
        val_str = '{:.02f}'.format(x) if pad else '{:.2g}'.format(x)
        if 0 <= np.abs(x) <= 1:
            if np.abs(x) == 0:
                return '.0'
            return val_str.replace("0", "", 1)
        else:
            return val_str

    major_formatter = FuncFormatter(xtick_formatter)
    major_formatter_nopad = FuncFormatter(functools.partial(xtick_formatter, pad=False))

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
    fig, ax = pyplot.subplots()
    for region, metric in itertools.product(regions, metrics):
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

    # neural + behavioral
    fields = ['neural: V4-neural_fit', 'neural: IT-neural_fit', 'i2n']
    err_fields = ['neural-error: V4-neural_fit', 'neural-error: IT-neural_fit', None]
    colors = ['V4', 'IT', 'behavior']
    add_ys = [0, 0, 0]  # [0.15, 0.15, 0.2]
    fig = pyplot.figure(figsize=(10, 5))
    axs = []
    # fig, axs = pyplot.subplots(1, 3, sharex=True, sharey=False, figsize=(10, 5))
    for i, (field, err_field, add_y, color) in enumerate(zip(fields, err_fields, add_ys, colors)):
        ax = fig.add_subplot(1, 3, i + 1, sharey=None if i != 1 else axs[0])
        axs.append(ax)
        # plot_field(ax=ax, data=data[basenets], field=field, err_field=err_field, label=field,
        #            color=score_color_mapping['basenet'], scatter_alpha=basenet_alpha, fit_method=None)
        plot_field(ax=ax, data=data[nonbasenets], field=field, err_field=err_field, label=field,
                   color=score_color_mapping[color], scatter_alpha=nonbasenet_alpha, fit_method=None)
        # ceiling
        ceiling = ceilings[field]
        ax.plot(ax.get_xlim(), [ceiling, ceiling],
                linestyle='dashed', linewidth=1., color=score_color_mapping['basenet'])
        # highlight
        for highlighted_model in highlighted_models:
            row = data[data['model'] == highlighted_model]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            x, y = row['performance'], row[field]
            highlight(ax, highlighted_model, x, y)
        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] + add_y])
        ax.set_title(color)
        ax.yaxis.set_major_formatter(major_formatter_nopad)
        clean_axis(ax)
        if i == 0:
            ax.set_ylabel('Brain-Score')
        if i == 1:
            for tk in ax.get_yticklabels():
                tk.set_visible(False)
        if i == 2:
            ax.yaxis.tick_right()

        # curve fit
        x, y = data[nonbasenets]['x'], data[nonbasenets][field]
        indices = np.argsort(x)
        x, y = np.array(x)[indices], np.array(y)[indices]
        z = np.polyfit(x, y, 2)
        f = np.poly1d(z)
        y_fit = f(x)
        # ax.plot(x, y_fit, color=score_color_mapping[color], linewidth=2., linestyle='dashed')
    ax = fig.add_subplot(111, frameon=False)
    ax.grid('off')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel('Imagenet performance (% top-1)', labelpad=5)

    pyplot.tight_layout()
    # fig.text(0.5, 0.01, 'Imagenet performance (% top-1)', ha='center')
    figs['subplots'] = fig

    # global
    fig, ax = pyplot.subplots()
    plot_global(ax, data[basenets], fit_method, color=score_color_mapping['basenet'], scatter_alpha=basenet_alpha)
    plot_global(ax, data[nonbasenets], fit_method, scatter_alpha=nonbasenet_alpha)
    # highlight
    for highlighted_model in highlighted_models:
        row = data[data['model'] == highlighted_model]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        x, y = row['performance'], row['global']
        highlight(ax, highlighted_model, x, y)
    post(ax, formatter=None)  # major_formatter_nopad)
    ax.set_ylabel('Mean BrainScore')
    figs['global'] = fig

    # rank
    fig, ax = pyplot.subplots()
    plot_rank(ax, data[basenets], fit_method, color=score_color_mapping['basenet'], scatter_alpha=basenet_alpha)
    plot_rank(ax, data[nonbasenets], fit_method, scatter_alpha=nonbasenet_alpha)
    # highlight
    for highlighted_model in highlighted_models:
        row = data[data['model'] == highlighted_model]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        x, y = row['performance'], row['rank']
        highlight(ax, highlighted_model, x, y)
    # SOTA: curve fit
    x, y = data['x'], data['rank']
    indices = np.argsort(x)
    x, y = np.array(x)[indices], np.array(y)[indices]
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)
    y_fit = f(x)
    # ax.plot(x, y_fit, color=score_color_mapping['basenet'], linewidth=2., linestyle='dashed')

    ax.invert_yaxis()
    ax.set_ylabel('$\overline{Rank}$')
    ax.set_yticklabels(['' if x < 0 else '{:.0f}'.format(x) if x != 0 else '1' for x in ax.get_yticks()])
    post(ax, formatter=None)  # major_formatter_nopad)
    figs['rank'] = fig

    # global-zoom
    zoom = data[nonbasenets]
    zoom = zoom[zoom['performance'] > 70]
    fig, ax = pyplot.subplots()
    plot_global(ax, zoom, fit_method, marker_size=150, scatter_alpha=0.7)
    # highlight
    for highlighted_model in highlighted_models:
        row = data[nonbasenets][data[nonbasenets]['model'] == highlighted_model]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        x, y = row['performance'], row['global']
        highlight(ax, highlighted_model, x, y)
    ax.set_ylabel('Mean BrainScore')
    post(ax)
    figs['global-zoom'] = fig

    return figs


def highlight(ax, highlighted_model, x, y):
    dx, dy = sum(ax.get_xlim()) * 0.02, sum(ax.get_ylim()) * 0.02
    # ax.arrow(x, y, dx, dy, head_width=0, color='black',
    #          head_starts_at_zero=True, length_includes_head=True)
    ax.plot([x, x + dx], [y, y + dy], color='black', linewidth=1.)
    ax.text(x + dx, y + dy, highlighted_model, fontsize=10)


def plot_field(ax, data, field, label, err_field=None, **kwargs):
    x = data['x']
    y = data[field]
    err = None if err_field is None else data[err_field]
    return _plot_score(x, y, error=err, label=label, ax=ax, fit_kwargs=dict(linestyle='dashed'),
                       **kwargs)


def plot_global(ax, data, fit_method, **kwargs):
    x = data['x']
    y = data['global'].values
    return _plot_score(x, y, label='global', fit_method=fit_method, label_long='BrainRank', ax=ax, **kwargs)


def plot_rank(ax, data, fit_method, **kwargs):
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
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # logging.DEBUG)
    fit = None

    models = get_models()
    missing_models = set(model_behavior.keys()) - set(models)
    print("Missing models:", missing_models)

    runs = {'performance': Mode.PERFORMANCE_VS_SCORE, 'depth': Mode.DEPTH_VS_SCORE, 'nonlins': Mode.NONLINS_VS_SCORE,
            'flops': Mode.FLOPS_VS_SCORE, 'numparams': Mode.PARAMS_VS_SCORE}
    for label, mode in runs.items():
        figs = plot_all(models, mode=mode, fit_method=fit, logx=label not in ['performance'])
        for name, fig in figs.items():
            savepath = 'results/scores/{}-{}.svg'.format(label, name)
            fig.savefig(savepath, format='svg')
            savepath = 'results/scores/{}-{}.pdf'.format(label, name)
            fig.savefig(savepath)
            print("Saved to", savepath)
        break
