import glob
import itertools
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import seaborn
from collections import defaultdict

from brainscore.utils import fullname
from matplotlib import pyplot

import candidate_models
from caching import cache
from candidate_models import score_physiology

seaborn.set()
seaborn.set_context("poster")
seaborn.set_style("whitegrid")

score_color_mapping = {
    'basenet': 'gray',
    'V4': '#00cc66',
    'IT': '#ff3232',
    'neural': '#000099',
    'behavior': '#bb9000',
    'performance': '#444444',
    'global': '#780ece'
}


def shaded_errorbar(x, y, error, ax=None, alpha=0.4, **kwargs):
    ax = ax or pyplot.gca()
    line = ax.plot(x, y, **kwargs)
    ax.fill_between(x, y - error, y + error, alpha=alpha, **kwargs)
    return line


def get_models():
    models = [file for file in glob.glob(os.path.join(os.path.dirname(__file__), '..', '..',
                                                      'output', 'candidate_models._score_physiology', '*'))]
    models = [re.match('.*/model=(.*),weights.*', file) for file in models]
    models = [match.group(1) for match in models if match]
    models = np.unique(models)
    potentially_broken_models = ['resnet-50_v1', 'resnet-101_v1', 'resnet-152_v1']
    models = [model for model in models if model not in potentially_broken_models]
    return models


def clean_axis(ax):
    ax.grid(b=True, which='major', linewidth=0.5)
    seaborn.despine(right=True)


class DataCollector(object):
    @cache()
    def __call__(self, neural_data=candidate_models.Defaults.neural_data):
        models = self.get_models()

        # neural scores
        data = self.parse_neural_data(models, neural_data)
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
                                                     'output', 'candidate_models._score_physiology', '*'))]
        models = [re.match('.*/model=(.*),weights.*', file) for file in models]
        models = [match.group(1) for match in models if match]
        models = np.unique(models)

        # check if all models were run
        all_models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'implementations', 'models.csv')
        all_models = pd.read_csv(all_models_path)['model'].values
        missing_models = set(all_models) - set(models)
        print("Missing models:", " ".join(missing_models))

        # remove models without metadata / broken models
        nometa_models = [model for model in models if model not in all_models]
        print("Removing models without metadata: ", " ".join(nometa_models))
        models = list(set(models) - set(nometa_models))
        potentially_broken_models = ['resnet-50_v1', 'resnet-101_v1', 'resnet-152_v1']
        models = [model for model in models if model not in potentially_broken_models]
        return models

    def parse_neural_data(self, models, neural_data=candidate_models.Defaults.neural_data):
        savepath = os.path.join(os.path.dirname(__file__), 'neural.csv')
        if os.path.isfile(savepath):
            return pd.read_csv(savepath)

        metrics = ['neural_fit']
        data = defaultdict(list)
        for model, metric in itertools.product(models, metrics):
            neural_score = score_physiology(model=model, neural_data=neural_data, metric_name=metric)
            neural_score = neural_score.aggregation
            aggregation_dims = ['aggregation', 'region']  # TODO: make generic to account for time
            assert all(dim in neural_score.dims for dim in aggregation_dims)
            reduce_dims = [dim for dim in neural_score.dims if dim not in aggregation_dims]
            # TODO: this just takes the maximum error but not necessarily the one corresponding to the maximum score
            neural_score = neural_score.max(reduce_dims)
            np.testing.assert_array_equal(neural_score.dims, aggregation_dims)

            data['model'].append(model)
            for region in np.unique(neural_score['region']):
                region_score = neural_score.sel(region=region)
                data[region].append(region_score.sel(aggregation='center').values)
                data[f"{region}-error"] = region_score.sel(aggregation='error').values
            data['neural_metric'].append(metric)

        data = pd.DataFrame(data=data)
        data.to_csv(savepath)
        return data

    def compute_brainscore(self, data):
        # method 1: mean everything
        global_scores = [[row['V4'], row['IT'], row['behavior']] for _, row in data.iterrows()]
        return np.mean(global_scores, axis=1)
        # method 2: mean(mean(v4, it), behavior)
        neural_scores = [[row['V4'], row['IT']] for _, row in data.iterrows()]
        neural_scores = np.mean(neural_scores, axis=1)
        global_scores = [[neural_score, row['behavior']] for (_, row), neural_score in zip(data.iterrows(), neural_scores)]
        return np.mean(global_scores, axis=1)

    def __repr__(self):
        return fullname(self)


def is_basenet(model_name):
    return model_name.startswith('basenet')
