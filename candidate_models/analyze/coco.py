import logging
import sys

import pandas as pd
import scipy.stats
import seaborn
from matplotlib import pyplot

from brainscore import benchmarks
from brainscore.metrics.transformations import CrossValidation, subset
from candidate_models import score_model
from candidate_models.analyze import DataCollector, align


def main(benchmark_name='coco', threshold=None, save=False):
    old_scores = DataCollector()()
    old_scores = old_scores[old_scores['benchmark'] == 'dicarlo.Majaj2015.IT']
    old_scores = old_scores[~old_scores['model'].isin(['cornet_z', 'cornet_r', 'cornet_r2'])]

    data = DataCollector()()
    benchmark_identifier = f'dicarlo.Kar2018{benchmark_name}'
    coco_data = data[data['benchmark'] == benchmark_identifier]

    if threshold is not None:
        benchmark = benchmarks.load(benchmark_identifier)
        ceiling = benchmark.ceiling.raw
        ceiling = CrossValidation().aggregate(ceiling)
        good_neuroids = ceiling.sel(aggregation='center') >= threshold
        good_neuroids = ceiling[{'neuroid': good_neuroids.values}]

        def coco_score(model):
            score = score_model(model, benchmark=benchmark_identifier)
            score = subset(score.raw, good_neuroids, subset_dims=['neuroid_id'])
            score = CrossValidation().aggregate(score.median('neuroid'))

            argmax = score.sel(aggregation='center').argmax('layer')
            max_score = score[{'layer': argmax.values}]
            return {'model': model, 'layer': max_score['layer'].values,
                    'score': max_score.sel(aggregation='center').values,
                    'error': max_score.sel(aggregation='error').values}

        coco_data = pd.DataFrame([coco_score(model) for model in coco_data['model']])
        coco_data.to_csv(f'results/supplement/{benchmark_name}-thresholded.csv')

    fig, ax = pyplot.subplots()
    coco_data = align(coco_data, old_scores, on='model')

    for with_cornet in [False, True]:
        current_old_scores = old_scores[[model.startswith('cornet') == with_cornet for model in old_scores['model']]]
        current_coco_data = align(coco_data, current_old_scores, on='model')
        x, xerr = current_old_scores['score'], current_old_scores['error']
        y, yerr = current_coco_data['score'], current_coco_data['error']
        color = '#808080' if not with_cornet else '#D4145A'
        ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, linestyle=' ', marker='.', color=color, ecolor=color)

    r, p = scipy.stats.pearsonr(old_scores['score'], coco_data['score'])
    assert p <= .05
    ax.text(ax.get_xlim()[1] - .015, ax.get_ylim()[0] + .005, f"r={r:.2f}")
    ax.set_xlabel('IT score, original neurons', fontsize=20)
    ax.set_ylabel(f"IT score, new neurons", fontsize=20)
    seaborn.despine(ax=ax, right=True, top=True)
    if save:
        pyplot.savefig(f'results/{benchmark_name}.png')
    return fig


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main('coco', threshold=.9, save=True)
    main('hvm', save=True)
