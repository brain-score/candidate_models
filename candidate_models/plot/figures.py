import itertools
import logging
import os
import re
import sys
from collections import defaultdict
from glob import glob
from typing import Union

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot

import candidate_models
from caching import cache
from candidate_models import score_physiology

seaborn.set()
seaborn.set_style("whitegrid")


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
        global_scores = [[neural_score, row['i2n']] for (_, row), neural_score in zip(data.iterrows(), neural_scores)]
        return np.mean(global_scores, axis=1)


def is_basenet(model_name):
    return model_name.startswith('basenet')


class Plot(object):
    def __call__(self, ax=None):
        data = self.collect_results()
        ax_given = ax is not None
        if not ax_given:
            fig, ax = self._create_fig()
            fig.tight_layout()
        self.apply(data, ax=ax)
        return fig if not ax_given else None

    def _create_fig(self):
        return pyplot.subplots(figsize=(10, 5))

    def apply(self, data, ax):
        raise NotImplementedError()

    def collect_results(self):
        data = DataCollector()()
        return data


class BrainScorePlot(Plot):
    def __init__(self):
        super(BrainScorePlot, self).__init__()
        self._nonbasenet_color = '#780ece'
        self._basenet_color = 'gray'

        self._nonbasenet_alpha = .7
        self._basenet_alpha = 0.3

    def _create_fig(self):
        return pyplot.subplots(figsize=(10, 8))

    def apply(self, data, ax):
        x = data['performance'].values
        y = data['brain-score'].values
        color = [self._nonbasenet_color if not is_basenet(model) else self._basenet_color
                 for model in data['model']]
        alpha = [self._nonbasenet_alpha if not is_basenet(model) else self._basenet_alpha
                 for model in data['model']]
        self.plot(x=x, y=y, color=color, alpha=alpha, ax=ax)
        ax.set_xlabel('Imagenet performance (% top-1)')
        ax.set_ylabel('Brain-Score')

    def plot(self, x, y, ax, error=None, label=None, color=None, marker_size=20, alpha: Union[float, list] = 0.3):
        def _plot(_x, _y, _error, plot_alpha):
            # if alpha is a list, provide a way to plot every point separately
            ax.scatter(_x, _y, label=label, color=color, alpha=plot_alpha, s=marker_size)
            if error:
                ax.errorbar(_x, _y, _error, label=label, color=color, alpha=plot_alpha,
                            elinewidth=1, linestyle='None')

        if isinstance(alpha, float):
            _plot(x, y, error, alpha)
        else:
            for _x, _y, _error, _alpha in zip(x, y, error if error is not None else [None] * len(x), alpha):
                _plot(_x, _y, _error, _alpha)


class BrainScoreZoomPlot(BrainScorePlot):
    def apply(self, data, ax):
        data = data[data['performance'] > 70]
        super(BrainScoreZoomPlot, self).apply(data, ax)

    def plot(self, x, y, ax, error=None, label=None, color=None, marker_size=100, alpha: Union[float, list] = 0.3):
        super(BrainScoreZoomPlot, self).plot(x=x, y=y, ax=ax, error=error, label=label,
                                             color=color, marker_size=marker_size, alpha=alpha)


class IndividualPlot(Plot):
    def collect_results(self):
        data = super().collect_results()
        data = data[data.apply(lambda row: not is_basenet(row['model']), axis=1)]
        return data

    def apply(self, data, ax):
        x, y, error = self.get_xye(data)
        self._plot(x=x, y=y, error=error, ax=ax)

        # TODO: ceiling
        # ax.plot(ax.get_xlim(), [ceiling, ceiling],
        #         linestyle='dashed', linewidth=1., color=score_color_mapping['basenet'])

        ax.grid(b=True, which='major', linewidth=0.5)
        self._despine(ax)

    def _despine(self, ax):
        seaborn.despine(ax=ax, top=True, right=True)

    def get_xye(self, data):
        raise NotImplementedError()

    def _plot(self, x, y, ax, error=None, alpha=0.5, **kwargs):
        ax.scatter(x, y, alpha=alpha, **kwargs)
        if error is not None:
            ax.errorbar(x, y, error, elinewidth=1, linestyle='None', alpha=alpha, **kwargs)


class V4Plot(IndividualPlot):
    def apply(self, data, ax):
        super(V4Plot, self).apply(data, ax)
        ax.set_ylabel('Neural Predictivity')

    def get_xye(self, data):
        return data['performance'], data['V4'], data['V4-error']

    def _plot(self, *args, **kwargs):
        super(V4Plot, self)._plot(*args, **kwargs, color='#00cc66')


class ITPlot(IndividualPlot):
    def apply(self, data, ax):
        super(ITPlot, self).apply(data, ax)
        for tk in ax.get_yticklabels():
            tk.set_visible(False)

    def get_xye(self, data):
        return data['performance'], data['IT'], data['IT-error']

    def _plot(self, *args, **kwargs):
        super(ITPlot, self)._plot(*args, **kwargs, color='#ff3232')


class BehaviorPlot(IndividualPlot):
    def apply(self, data, ax):
        super(BehaviorPlot, self).apply(data, ax)
        ax.yaxis.tick_right()
        ax.set_ylabel('Behavioral Predictivity', rotation=270, labelpad=15)
        ax.yaxis.set_label_position("right")

    def _despine(self, ax):
        seaborn.despine(ax=ax, left=True, top=True, right=False)
        ax.tick_params(axis='y', which='both', length=0)

    def get_xye(self, data):
        return data['performance'], data['behavior'], None

    def _plot(self, *args, **kwargs):
        super(BehaviorPlot, self)._plot(*args, **kwargs, color='#bb9000')


class IndividualPlots(object):
    def __call__(self):
        fig = pyplot.figure(figsize=(10, 3))
        self.apply(fig)
        fig.tight_layout()
        return fig

    def apply(self, fig):
        plotters = [V4Plot(), ITPlot(), BehaviorPlot()]
        axes = []
        for i, plotter in enumerate(plotters):
            ax = fig.add_subplot(1, 3, i + 1, sharey=None if i != 1 else axes[0])
            axes.append(ax)
            plotter(ax=ax)

        # joint xlabel
        ax = fig.add_subplot(111, frameon=False)
        ax.grid('off')
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax.set_xlabel('Imagenet performance (% top-1)', labelpad=5)


class PaperFigures(object):
    def __init__(self):
        self._savedir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        self._save_formats = ['svg', 'pdf']

    def __call__(self):
        figs = {
            'brain-score': BrainScorePlot(),
            'brain-score-zoom': BrainScoreZoomPlot(),
            'individual': IndividualPlots(),
        }
        for name, fig_maker in figs.items():
            fig = fig_maker()
            self.save(fig, name)

    def save(self, fig, name):
        for extension in self._save_formats:
            savepath = os.path.join(self._savedir, f"{name}.{extension}")
            fig.savefig(savepath, format=extension)
            print("Saved to", savepath)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plotter = PaperFigures()
    plotter()
