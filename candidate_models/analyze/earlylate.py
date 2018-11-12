import logging
import sys

from matplotlib import pyplot

from candidate_models.analyze import DataCollector


def main():
    data = DataCollector()()
    data = data[data['benchmark'] == 'dicarlo.Majaj2015.earlylate']

    fig, axes = pyplot.subplots(ncols=2)
    for region, ax in zip(['V4', 'IT'], axes):
        early_scores = data[(data['region'] == region) & (data['time_bin_start'] == 90)]
        late_scores = data[(data['region'] == region) & (data['time_bin_start'] == 190)]
        late_scores = _align(late_scores, early_scores, on='model')
        x, xerr = early_scores['score'], early_scores['error']
        y, yerr = late_scores['score'], late_scores['error']
        colors = ['b' if model != 'cornet_s' else 'r' for model in early_scores['model']]
        ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, linestyle=' ', marker='.', ecolor=colors)
        ax.set_xlabel('early (90-110)')
        ax.set_ylabel('late (190-210)')
        ax.set_title(region)
    pyplot.savefig('results/earlylate.png')


def _align(data1, data2, on):
    data1 = data1[data1[on].isin(data2[on])]
    data1 = data1.set_index(on).reindex(index=data2[on]).reset_index()
    return data1


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
