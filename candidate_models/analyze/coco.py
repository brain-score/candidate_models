import itertools

import scipy.stats
import logging
import os
import re
import sys
from collections import defaultdict
from glob import glob

import pandas as pd
from matplotlib import pyplot

from brainscore import benchmarks
from brainscore.metrics.transformations import CrossValidation, subset
from candidate_models import score_model
from candidate_models.analyze import DataCollector, align
from result_caching import store


def main(benchmark_name='coco', threshold=None):
    old_scores = collect_old_scores(os.path.join(os.path.dirname(__file__), '..', '..',
                                                 'output', 'candidate_models._score_physiology'))
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


@store()
def collect_old_scores(results_path):
    score_files = glob(os.path.join(results_path, '*.pkl'))
    regex = re.compile(r"""model=(?P<model>[^,]*),
                           weights=imagenet,pca_components=1000,
                           neural_data=dicarlo.Majaj2015,target_splits=\('region',\),metric_name=neural_fit""", re.X)
    data = defaultdict(list)
    for file in score_files:
        meta = re.search(regex, file)
        if not meta:
            continue
        model = meta.group('model')
        assert model not in data['model']
        score = pd.read_pickle(file)['data'].aggregation.sel(ceiled=False)
        for region in ['V4', 'IT']:
            data['model'].append(model)
            data['benchmark'].append(f'dicarlo.Majaj2015.{region}')
            region_score = score.sel(region=region)
            argmax = region_score.sel(aggregation='center').argmax('layer')
            max_score = region_score[{'layer': argmax.values}]
            data['score'].append(max_score.sel(aggregation='center').values)
            data['error'].append(max_score.sel(aggregation='error').values)
            data['layer'].append(max_score['layer'].values)
    data = pd.DataFrame(data)
    potentially_broken_models = ['resnet-50_v1', 'resnet-101_v1', 'resnet-152_v1']
    data = data[~data['model'].isin(potentially_broken_models)]
    return data


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main('coco', threshold=.9)
    main('hvm')
