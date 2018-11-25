import logging
import sys

import pandas as pd
import scipy.stats
from matplotlib import pyplot

from brainscore import benchmarks
from brainscore.metrics.transformations import CrossValidation, subset
from candidate_models import score_model
from candidate_models.analyze import DataCollector, align


def main(benchmark_name='coco', threshold=None):
    old_scores = DataCollector()()
    old_scores = old_scores[old_scores['benchmark'] == 'dicarlo.Majaj2015.IT']

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

    coco_data = align(coco_data, old_scores, on='model')

    fig, ax = pyplot.subplots()
    x, xerr = old_scores['score'], old_scores['error']
    y, yerr = coco_data['score'], coco_data['error']
    r, p = scipy.stats.pearsonr(x, y)
    assert p <= .05
    colors = ['b' if not model.startswith('cornet') else 'r' for model in coco_data['model']]
    ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, linestyle=' ', marker='.', ecolor=colors)
    ax.text(ax.get_xlim()[1] - .01, ax.get_ylim()[0], f"r={r:.2f}")
    ax.set_xlabel('Ha hvm IT')
    ax.set_ylabel(f"Ko's {benchmark_name}")
    pyplot.savefig(f'results/{benchmark_name}.png')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main('coco', threshold=.9)
    main('hvm')
