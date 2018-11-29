import itertools
import os
import pickle
import re
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot

from brainscore.assemblies import merge_data_arrays
from result_caching import cache
from candidate_models import score_model, model_layers


def shaded_errorbar(x, y, error, ax=None, alpha=0.4, shaded_kwargs=None, **kwargs):
    shaded_kwargs = shaded_kwargs or {}
    ax = ax or pyplot.gca()
    line = ax.plot(x, y, **kwargs)
    ax.fill_between(x, y - error, y + error, alpha=alpha, **shaded_kwargs)
    return line


def clean_axis(ax):
    ax.grid(b=True, which='major', linewidth=0.5)
    seaborn.despine(right=True)


class DataCollector:
    @cache()
    def __call__(self):
        # neural scores
        data = self.parse_neural_scores()
        data = data[~data['model'].isin(['cornet_z', 'cornet_r', 'cornet_r2'])]
        # concat ImageNet performance
        model_meta = self._get_models_meta()
        for score_field, benchmark in [('performance', 'ImageNet')]:
            benchmark_data = model_meta[['model', score_field]]
            benchmark_data.rename(columns={score_field: 'score'}, inplace=True)
            benchmark_data['benchmark'] = [benchmark] * len(benchmark_data)
            data = pd.concat([data, benchmark_data])
        data['score'][data['benchmark'] == 'ImageNet'] = 100 * data['score'][data['benchmark'] == 'ImageNet']
        # basenets
        basenets = self._get_basenets()
        data = pd.concat([data, basenets])
        # behavior
        behavior = self._get_behavior()
        data = pd.concat([data, behavior])
        # brain-score
        brain_score = self.compute_brainscore(data)
        brain_score['benchmark'] = ['Brain-Score'] * len(brain_score)
        data = pd.concat([data, brain_score])
        # attach meta
        model_meta = model_meta[['model', 'link', 'bibtex']]
        data = data.merge(model_meta, on='model')
        # filter broken
        potentially_broken_models = ['resnet-50_v1', 'resnet-101_v1', 'resnet-152_v1']
        data = data[~data['model'].isin(potentially_broken_models)]
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

    def _get_behavior(self):
        filepath = os.path.join(os.path.dirname(__file__), '..', 'models', 'implementations', 'behavior.csv')
        behavior = pd.read_csv(filepath)
        behavior = behavior[['model', 'benchmark', 'score']]
        behavior['layer'] = self._get_behavioral_layers(behavior['model'])
        return behavior

    def _get_basenets(self):
        results = []
        for filepath in glob(os.path.join(os.path.dirname(__file__), '..', '..',
                                          'output', 'backup', 'neurality.score_physiology', '*.pkl')):
            model = re.search(r'model=([^,]+),', filepath)
            assert model
            model = model.group(1)
            if not model.startswith('basenet'):
                continue  # reject
            with open(filepath, 'rb') as f:
                score = pickle.load(f)
            score = score['data']
            center, error = score.center, score.error
            center = center.expand_dims('aggregation')
            center['aggregation'] = ['center']
            error = error.expand_dims('aggregation')
            error['aggregation'] = ['error']
            score = merge_data_arrays([center, error])
            for region in ['V4', 'IT']:
                region_score = score.sel(region=region)
                argmax = region_score.sel(aggregation='center').argmax('layer')
                region_score = region_score[{'layer': argmax.values}]

                results.append({'model': model,
                                'benchmark': f'dicarlo.Majaj2015.{region}',
                                'score': region_score.sel(aggregation='center').values,
                                'error': region_score.sel(aggregation='error').values,
                                })
        results = pd.DataFrame(results)

        behavior = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'basenets_correct.pkl'))
        assert all([model.startswith('basenet') for model in behavior['model']])
        behavior = behavior.rename(columns={'r': 'score'})
        behavior['benchmark'] = 'dicarlo.Rajalingham2018'
        behavior = behavior[['model', 'benchmark', 'score']]
        behavior['layer'] = self._get_behavioral_layers(behavior['model'])
        results = pd.concat([results, behavior])

        results.to_csv(os.path.join(os.path.dirname(__file__), 'basenets.csv'))
        return results

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

            missing_models = set(models) - set(model_layers.keys())
            print(f"missing from {self._benchmark}: {missing_models}")
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

    class TemporalScoreParser(DividingScoreParser):
        def append_score(self, data, score, model):
            divider_values = {divider: np.unique(score[divider]) for divider in self._dividers}
            for key_values in (dict(zip(divider_values.keys(), values))
                               for values in itertools.product(*divider_values.values())):
                divided_score = score.sel(**key_values)

                divided_score = divided_score.expand_dims('layer')
                divided_score['layer'] = ["IT-layer"]

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
            DataCollector.DividingScoreParser('dicarlo.Majaj2015.earlylate-alternatives', dividers=[
                'region', 'time_bin_start']),
            DataCollector.ScoreParser('dicarlo.Kar2018coco'),
            DataCollector.ScoreParser('dicarlo.Kar2018hvm'),
            DataCollector.TemporalScoreParser('dicarlo.Majaj2015.temporal-mean.IT', dividers=['time_slice']),
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
            brain_score = np.mean([model_rows['score'][model_rows['benchmark'] == benchmark].values
                                   for benchmark in benchmarks])
            return brain_score
            # method 2: mean(mean(v4, it), behavior)
            neural_scores = [[row['V4'], row['IT']] for _, row in data.iterrows()]
            neural_scores = np.mean(neural_scores, axis=1)
            brain_score = [[neural_score, row['behavior']] for (_, row), neural_score in
                           zip(data.iterrows(), neural_scores)]
            return np.mean(brain_score, axis=1)

        brainscores = data.groupby('model').apply(compute)
        brainscores = brainscores.dropna()
        return pd.DataFrame({'model': brainscores.index.values, 'score': brainscores.values})


def filter_basenets(data, include=True):
    basenet_filter = [is_basenet(row['model']) for _, row in data.iterrows()]
    if include:
        return data[basenet_filter]
    else:
        return data[[not basenet for basenet in basenet_filter]]


def is_basenet(model_name):
    return model_name.startswith('basenet')


def align(data1, data2, on):
    data1 = data1[data1[on].isin(data2[on])]
    data1 = data1.set_index(on).reindex(index=data2[on]).reset_index()
    return data1
