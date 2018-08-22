import glob
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot

from caching import cache
from candidate_models import score_physiology


def shaded_errorbar(x, y, error, ax=None, alpha=0.4, **kwargs):
    ax = ax or pyplot.gca()
    line = ax.plot(x, y, **kwargs)
    ax.fill_between(x, y - error, y + error, alpha=alpha, **kwargs)
    return line


def clean_axis(ax):
    ax.grid(b=True, which='major', linewidth=0.5)
    seaborn.despine(right=True)


class DataCollector(object):
    @cache()
    def __call__(self):
        models = self.get_models()

        # neural scores
        data = self.parse_neural_scores(models)
        # merge with behavior, performance and meta
        meta_filepath = os.path.join(os.path.dirname(__file__), '..', 'models', 'implementations', 'models.csv')
        model_meta = pd.read_csv(meta_filepath)
        model_meta = model_meta.rename(columns={'behav_r': 'behavior', 'top1': 'performance'})
        data = data.merge(model_meta[['model', 'behavior', 'performance', 'link', 'bibtex']], on='model')
        data['performance'] = 100 * data['performance']
        # brain-score
        data['brain-score'] = self.compute_brainscore(data)
        # rank
        data['rank'] = data['brain-score'].rank(ascending=False)
        return data

    def get_models(self):
        models = [file for file in glob(os.path.join(os.path.dirname(__file__), '..', '..',
                                                     'output', 'brainscore.benchmarks.ToliasCadena2017._call', '*'))]
        models = [re.match('.*/identifier=(.*)\.pkl', file) for file in models]
        models = [match.group(1) for match in models if match]
        models = np.unique(models)

        # check if all models were run
        all_models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'implementations', 'models.csv')
        all_models = pd.read_csv(all_models_path)['model'].values
        missing_models = set(all_models) - set(models)
        print("Missing basenets:", " ".join([model for model in missing_models if model.startswith('basenet')]))
        print("Missing non-basenets:", " ".join([model for model in missing_models if not model.startswith('basenet')]))

        # remove models without metadata / broken models
        nometa_models = [model for model in models if model not in all_models]
        print("Removing models without metadata: ", " ".join(nometa_models))
        models = list(set(models) - set(nometa_models))
        potentially_broken_models = ['resnet-50_v1', 'resnet-101_v1', 'resnet-152_v1']
        models = [model for model in models if model not in potentially_broken_models]
        return models

    class ScoreParser(object):
        def __init__(self, benchmark):
            self._benchmark = benchmark

        def __call__(self, models):
            data = defaultdict(list)
            for model in models:
                data['model'].append(model)
                ceiled_score, score = score_physiology(model=model, benchmark=self._benchmark, return_unceiled=True)
                score = score.aggregation
                # TODO: this just takes the maximum error but not necessarily the one corresponding to the maximum score
                score = score.max('layer')
                self._parse_score(score, data)
                # bs .435 | alexnet | top-1 57.67 | V4 .663 | IT .587 | behavior .245

            return pd.DataFrame(data)

        def _parse_score(self, model, target):
            raise NotImplementedError()

        def _set_score(self, target, label, score):
            center, err = score.sel(aggregation='center').values, score.sel(aggregation='error').values
            assert center.size == err.size == 1
            target[label].append(center.tolist())
            target[f"{label}-error"].append(err.tolist())

    class DicarloMajaj2015Parser(ScoreParser):
        def __init__(self):
            super(DataCollector.DicarloMajaj2015Parser, self).__init__('dicarlo.Majaj2015')

        def _parse_score(self, score, target):
            for region in np.unique(score['region']):
                region_score = score.sel(region=region)
                self._set_score(target, region, score=region_score)

    class ToliasCadena2017Parser(ScoreParser):
        def __init__(self):
            super(DataCollector.ToliasCadena2017Parser, self).__init__('tolias.Cadena2017')

        def _parse_score(self, score, target):
            self._set_score(target, 'V1', score)

    def parse_neural_scores(self, models):
        savepath = os.path.join(os.path.dirname(__file__), 'neural.csv')
        if os.path.isfile(savepath):
            return pd.read_csv(savepath)

        benchmarks = {
            'dicarlo.Majaj2015': DataCollector.DicarloMajaj2015Parser(),
            'tolias.Cadena2017': DataCollector.ToliasCadena2017Parser()
        }
        data = None
        for benchmark, parser in benchmarks.items():
            benchmark_data = parser(models)
            data = benchmark_data if data is None else data.merge(benchmark_data, on='model')

        data.to_csv(savepath)
        return data

    def compute_brainscore(self, data):
        # method 1: mean everything
        global_scores = [[row['V4'], row['IT'], row['behavior']] for _, row in data.iterrows()]
        return np.mean(global_scores, axis=1)
        # method 2: mean(mean(v4, it), behavior)
        neural_scores = [[row['V4'], row['IT']] for _, row in data.iterrows()]
        neural_scores = np.mean(neural_scores, axis=1)
        global_scores = [[neural_score, row['behavior']] for (_, row), neural_score in
                         zip(data.iterrows(), neural_scores)]
        return np.mean(global_scores, axis=1)


def is_basenet(model_name):
    return model_name.startswith('basenet')
