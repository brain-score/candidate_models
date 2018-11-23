import itertools
import logging
import sys

import numpy as np
from matplotlib import pyplot
from ast import literal_eval
from brainscore import benchmarks
from candidate_models.analyze import DataCollector, shaded_errorbar


def temporal_mean():
    benchmark = 'dicarlo.Majaj2015.temporal-mean.IT'
    data = DataCollector()()
    data = data[data['benchmark'] == benchmark]
    ceiling = benchmarks.load(benchmark).ceiling
    ceiling['time_bin_start'] = 'time_bin', [literal_eval(time_bin)[0] for time_bin in ceiling['time_bin'].values]
    ceiling['time_bin_end'] = 'time_bin', [literal_eval(time_bin)[1] for time_bin in ceiling['time_bin'].values]

    fig, ax = pyplot.subplots()
    ceiling = ceiling.sortby('time_bin_start')
    x = list(itertools.chain(*[(start, end) for (start, end) in zip(
        ceiling['time_bin_start'].values, ceiling['time_bin_end'].values)]))
    y, yerr = ceiling.sel(aggregation='center').values, ceiling.sel(aggregation='error').values
    y, yerr = np.repeat(y, 2), np.repeat(yerr, 2)
    shaded_errorbar(x=x, y=y, error=yerr,
                    ax=ax, color='gray', label='ceiling')
    colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']  # https://stackoverflow.com/a/42091037/2225200
    for model, color in zip(np.unique(data['model']), colors):
        model_data = data[data['model'] == model]
        model_data['time_slice'] = [literal_eval(time_slice.replace(')', '),'))
                                    for time_slice in model_data['time_slice']]
        model_data['time_bin_start'] = [time_slice[0] for time_slice in model_data['time_slice']]
        model_data = model_data.sort_values('time_bin_start')
        time_slices = list(itertools.chain(*[(time_slice[0][0], time_slice[-1][-1])
                                             for time_slice in model_data['time_slice']]))
        y, yerr = np.repeat(model_data['score'].values, 2), np.repeat(model_data['error'].values, 2)
        shaded_errorbar(x=time_slices, y=y, error=yerr, label=model, color=color, shaded_kwargs=dict(color=color))
    ax.set_xlabel('time')
    ax.set_ylabel('pearson r')
    ax.legend()
    pyplot.savefig(f'results/temporal-{benchmark}.png')


def temporal_bin():
    benchmark = 'dicarlo.Majaj2015.temporal.IT'
    data = DataCollector()()
    data = data[data['benchmark'] == benchmark]
    ceiling = benchmarks.load(benchmark).ceiling
    ceiling['time_bin_start'] = 'time_bin', [literal_eval(time_bin)[0] for time_bin in ceiling['time_bin'].values]
    ceiling['time_bin_end'] = 'time_bin', [literal_eval(time_bin)[1] for time_bin in ceiling['time_bin'].values]

    fig, ax = pyplot.subplots()
    ceiling = ceiling.sortby('time_bin_start')
    x = list(itertools.chain(*[(start, end) for (start, end) in zip(
        ceiling['time_bin_start'].values, ceiling['time_bin_end'].values)]))
    y, yerr = ceiling.sel(aggregation='center').values, ceiling.sel(aggregation='error').values
    y, yerr = np.repeat(y, 2), np.repeat(yerr, 2)
    shaded_errorbar(x=x, y=y, error=yerr,
                    ax=ax, color='gray', label='ceiling')
    colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']  # https://stackoverflow.com/a/42091037/2225200
    for model, color in zip(np.unique(data['model']), colors):
        model_data = data[data['model'] == model]
        model_data['time_bin'] = [literal_eval(time_slice) for time_slice in model_data['time_bin']]
        model_data['time_bin_start'] = [time_slice[0] for time_slice in model_data['time_bin']]
        model_data = model_data.sort_values('time_bin_start')
        time_bins = list(itertools.chain(*model_data['time_bin']))
        y, yerr = np.repeat(model_data['score'].values, 2), np.repeat(model_data['error'].values, 2)
        shaded_errorbar(x=time_bins, y=y, error=yerr, label=model, color=color, shaded_kwargs=dict(color=color))
    ax.set_xlabel('time')
    ax.set_ylabel('pearson r')
    ax.legend()
    pyplot.savefig(f'results/temporal-{benchmark}.png')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    temporal_bin()
