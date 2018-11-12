import itertools
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot

from result_caching import cache
from candidate_models import score_model, model_layers


def shaded_errorbar(x, y, error, ax=None, alpha=0.4, **kwargs):
    ax = ax or pyplot.gca()
    line = ax.plot(x, y, **kwargs)
    ax.fill_between(x, y - error, y + error, alpha=alpha, **kwargs)
    return line


def clean_axis(ax):
    ax.grid(b=True, which='major', linewidth=0.5)
    seaborn.despine(right=True)


class DataCollector:
    @cache()
    def __call__(self):
        # neural scores
        data = self.parse_neural_scores()
        # concat behavior, performance
        model_meta = self._get_models_meta()
        for score_field, benchmark in [('performance', 'ImageNet'), ('behavior', 'dicarlo.Rajalingham2018')]:
            benchmark_data = model_meta[['model', score_field]]
            benchmark_data.rename(columns={score_field: 'score'}, inplace=True)
            benchmark_data['error'] = [0] * len(benchmark_data)
            benchmark_data['benchmark'] = [benchmark] * len(benchmark_data)
            data = pd.concat([data, benchmark_data])
        # manual edits
        data['score'][data['benchmark'] == 'ImageNet'] = 100 * data['score'][data['benchmark'] == 'ImageNet']
        data['layer'][data['benchmark'] == 'dicarlo.Rajalingham2018'] = \
            self._get_behavioral_layers(data['model'][data['benchmark'] == 'dicarlo.Rajalingham2018'])
        # brain-score
        brain_score = self.compute_brainscore(data)
        brain_score['benchmark'] = ['Brain-Score'] * len(brain_score)
        data = pd.concat([data, brain_score])
        # attach meta
        model_meta = model_meta[['model', 'link', 'bibtex']]
        data = data.merge(model_meta, on='model')
        # rank
        for benchmark in np.unique(data['benchmark']):
            data['rank'] = data['score'][data['benchmark'] == benchmark].rank(ascending=False)
        return data

    def _get_models_meta(self):
        meta_filepath = os.path.join(os.path.dirname(__file__), '..', 'models', 'implementations', 'models.csv')
        model_meta = pd.read_csv(meta_filepath)
        model_meta = model_meta.rename(columns={'behav_r': 'behavior', 'top1': 'performance'})
        basenet_meta = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', '..', 'basenets_correct.pkl'))
        basenet_meta = basenet_meta.rename(columns={'r': 'behavior', 'top1': 'performance'})
        basenet_meta = basenet_meta[['model', 'behavior', 'performance']]
        model_meta = pd.concat([model_meta, basenet_meta])
        return model_meta

    class ScoreParser:
        def __init__(self, benchmark):
            self._benchmark = benchmark

        def __call__(self, storage_path=os.path.join(os.path.dirname(__file__), '..', '..',
                                                     'output', 'candidate_models._score_model')):
            models = self._find_models(storage_path)

            data = defaultdict(list)
            for model in models:
                score = score_model(model=model, benchmark=self._benchmark)
                self.append_score(data, score, model)

            data['benchmark'] = [self._benchmark] * len(data['score'])
            return pd.DataFrame(data)

        def append_score(self, data, score, model):
            data['model'].append(model)
            argmax = score.sel(aggregation='center').argmax('layer')
            max_score = score[{'layer': argmax.values}]
            data['score'].append(max_score.sel(aggregation='center').values)
            data['error'].append(max_score.sel(aggregation='error').values)
            data['layer'].append(max_score['layer'].values)

        def _find_models(self, storage_path):
            file_glob = os.path.join(storage_path, f'*benchmark_identifier={self._benchmark},*.pkl')
            models = list(glob(file_glob))
            models = [re.search('model_identifier=([^,]*)', file) for file in models]
            models = [match.group(1) for match in models if match]
            return set(models)

    class DividingScoreParser(ScoreParser):
        def __init__(self, *args, dividers, **kwargs):
            super(DataCollector.DividingScoreParser, self).__init__(*args, **kwargs)
            self._dividers = dividers if (isinstance(dividers, list) or isinstance(dividers, tuple)) else [dividers]

        def append_score(self, data, score, model):
            divider_values = {divider: np.unique(score[divider]) for divider in self._dividers}
            for key_values in (dict(zip(divider_values.keys(), values))
                               for values in itertools.product(*divider_values.values())):
                divided_score = score.sel(**key_values)
                super(DataCollector.DividingScoreParser, self).append_score(data=data, score=divided_score, model=model)
                for key, value in key_values.items():
                    data[key].append(value)

    def parse_neural_scores(self):
        savepath = os.path.join(os.path.dirname(__file__), 'neural.csv')
        if os.path.isfile(savepath):
            return pd.read_csv(savepath)

        benchmark_parsers = [
            DataCollector.ScoreParser('tolias.Cadena2017'),
            DataCollector.ScoreParser('movshon.FreemanZiemba2013.V1'),
            DataCollector.ScoreParser('movshon.FreemanZiemba2013.V2'),
            DataCollector.ScoreParser('dicarlo.Majaj2015.V4'),
            DataCollector.ScoreParser('dicarlo.Majaj2015.IT'),
            DataCollector.DividingScoreParser('dicarlo.Majaj2015.earlylate', dividers=['region', 'time_bin_start']),
        ]
        data = None
        for parser in benchmark_parsers:
            benchmark_data = parser()
            data = benchmark_data if data is None else pd.concat([data, benchmark_data])

        data.to_csv(savepath, index=False)
        return data

    def _get_behavioral_layers(self, model_names):
        return [model_layers[model][-1] for model in model_names]

    def compute_brainscore(self, data):
        def compute(model_rows):
            benchmarks = ['dicarlo.Majaj2015.V4', 'dicarlo.Majaj2015.IT', 'dicarlo.Rajalingham2018']
            occurrences = model_rows['benchmark'].value_counts()
            if not all(benchmark in occurrences for benchmark in benchmarks):
                return np.nan
            assert all(occurrences[benchmark] == 1 for benchmark in benchmarks)
            # method 1: mean everything
            brain_score = np.mean([data['score'][data['benchmark'] == benchmark] for benchmark in benchmarks])
            return np.mean(brain_score)
            # method 2: mean(mean(v4, it), behavior)
            neural_scores = [[row['V4'], row['IT']] for _, row in data.iterrows()]
            neural_scores = np.mean(neural_scores, axis=1)
            brain_score = [[neural_score, row['behavior']] for (_, row), neural_score in
                           zip(data.iterrows(), neural_scores)]
            return np.mean(brain_score, axis=1)

        brainscores = data.groupby('model').apply(compute)
        return pd.DataFrame({'model': brainscores.index.values, 'score': brainscores.values})


def filter_basenets(data, include=True):
    basenet_filter = [is_basenet(row['model']) for _, row in data.iterrows()]
    if include:
        return data[basenet_filter]
    else:
        return data[[not basenet for basenet in basenet_filter]]


def is_basenet(model_name):
    return model_name.startswith('basenet')
