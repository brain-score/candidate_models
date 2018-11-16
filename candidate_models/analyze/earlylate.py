import logging
import sys

import seaborn

seaborn.set()
from matplotlib import pyplot

from candidate_models.analyze import DataCollector, align


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

    data = DataCollector()()

    plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate'], early=(90, 110), late=(190, 210))
    plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate-alternatives'], early=(70, 90), late=(170, 190))
    plot(data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate-alternatives'], early=(70, 90), late=(150, 250))
