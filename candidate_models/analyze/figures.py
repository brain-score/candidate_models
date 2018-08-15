import logging
import os
import sys
from typing import Union

import seaborn
from matplotlib import pyplot

from candidate_models.analyze import DataCollector, is_basenet

seaborn.set()
seaborn.set_style("whitegrid")


class Plot(object):
    def __init__(self, highlighted_models=()):
        self._highlighted_models = highlighted_models

    def __call__(self, ax=None):
        data = self.collect_results()
        ax_given = ax is not None
        if not ax_given:
            fig, ax = self._create_fig()
        self.apply(data, ax=ax)
        self.highlight_models(ax, data)
        if not ax_given:
            fig.tight_layout()
            return fig
        return None

    def _create_fig(self):
        return pyplot.subplots(figsize=(10, 5))

    def apply(self, data, ax):
        raise NotImplementedError()

    def get_xye(self, data):
        raise NotImplementedError()

    def collect_results(self):
        data = DataCollector()()
        return data

    def highlight_models(self, ax, data):
        for highlighted_model in self._highlighted_models:
            row = data[data['model'] == highlighted_model]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            x, y, error = self.get_xye(row)
            self._highlight(ax, highlighted_model, x, y)

    def _highlight(self, ax, label, x, y):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        dx, dy = (xlim[1] - xlim[0]) * 0.02, (ylim[1] - ylim[0]) * 0.02
        ax.plot([x, x + dx], [y, y + dy], color='black', linewidth=1.)
        ax.text(x + dx, y + dy, label, fontsize=10)


class BrainScorePlot(Plot):
    def __init__(self, highlighted_models=()):
        super(BrainScorePlot, self).__init__(highlighted_models=highlighted_models)
        self._nonbasenet_color = '#780ece'
        self._basenet_color = 'gray'

        self._nonbasenet_alpha = .7
        self._basenet_alpha = 0.3

    def _create_fig(self):
        return pyplot.subplots(figsize=(10, 8))

    def get_xye(self, data):
        return data['performance'], data['brain-score'], None

    def apply(self, data, ax):
        x, y, error = self.get_xye(data)
        x, y = x.values, y.values
        color = [self._nonbasenet_color if not is_basenet(model) else self._basenet_color
                 for model in data['model']]
        alpha = [self._nonbasenet_alpha if not is_basenet(model) else self._basenet_alpha
                 for model in data['model']]
        self.plot(x=x, y=y, color=color, alpha=alpha, ax=ax)
        ax.set_xlabel('Imagenet performance (% top-1)')
        ax.set_ylabel('Brain-Score')

    def plot(self, x, y, ax, error=None, label=None, color=None, marker_size=50, alpha: Union[float, list] = 0.3):
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
    def __init__(self, ceiling, highlighted_models=()):
        super(IndividualPlot, self).__init__(highlighted_models=highlighted_models)
        self._ceiling = ceiling

    def collect_results(self):
        data = super().collect_results()
        data = data[data.apply(lambda row: not is_basenet(row['model']), axis=1)]
        return data

    def apply(self, data, ax):
        x, y, error = self.get_xye(data)
        self._plot(x=x, y=y, error=error, ax=ax)
        self.highlight_models(ax, data)
        ax.grid(b=True, which='major', linewidth=0.5)
        self._despine(ax)

    def _despine(self, ax):
        seaborn.despine(ax=ax, top=True, right=True)

    def _plot(self, x, y, ax, error=None, alpha=0.5, s=20, **kwargs):
        ax.scatter(x, y, alpha=alpha, s=s, **kwargs)
        if error is not None:
            ax.errorbar(x, y, error, elinewidth=1, linestyle='None', alpha=alpha, **kwargs)
        ax.plot(ax.get_xlim(), [self._ceiling, self._ceiling], linestyle='dashed', linewidth=1., color='gray')


class V4Plot(IndividualPlot):
    def __init__(self, highlighted_models=()):
        super(V4Plot, self).__init__(ceiling=.892, highlighted_models=highlighted_models)

    def apply(self, data, ax):
        super(V4Plot, self).apply(data, ax)
        ax.set_title('V4')
        ax.set_ylabel('Neural Predictivity')

    def get_xye(self, data):
        return data['performance'], data['V4'], data['V4-error']

    def _plot(self, *args, **kwargs):
        super(V4Plot, self)._plot(*args, **kwargs, color='#00cc66')


class ITPlot(IndividualPlot):
    def __init__(self, highlighted_models=()):
        super(ITPlot, self).__init__(ceiling=.817, highlighted_models=highlighted_models)

    def apply(self, data, ax):
        super(ITPlot, self).apply(data, ax)
        ax.set_title('IT')
        for tk in ax.get_yticklabels():
            tk.set_visible(False)

    def get_xye(self, data):
        return data['performance'], data['IT'], data['IT-error']

    def _plot(self, *args, **kwargs):
        super(ITPlot, self)._plot(*args, **kwargs, color='#ff3232')


class BehaviorPlot(IndividualPlot):
    def __init__(self, highlighted_models=()):
        super(BehaviorPlot, self).__init__(ceiling=.479, highlighted_models=highlighted_models)

    def apply(self, data, ax):
        super(BehaviorPlot, self).apply(data, ax)
        ax.set_title('Behavior')
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
    def __init__(self, highlighted_models=()):
        self._highlighted_models = highlighted_models

    def __call__(self):
        fig = pyplot.figure(figsize=(10, 3))
        self.apply(fig)
        fig.tight_layout()
        return fig

    def apply(self, fig):
        plotters = [
            V4Plot(highlighted_models=self._highlighted_models),
            ITPlot(highlighted_models=self._highlighted_models),
            BehaviorPlot(highlighted_models=self._highlighted_models)
        ]
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
        highlighted_models = [
            'pnasnet_large',  # good performance
            'resnet-152_v2', 'densenet-169',  # best overall
            'alexnet',
            'resnet-50_v2',  # good V4
            'mobilenet_v2_0.75_224',  # best mobilenet
            'mobilenet_v1_1.0.224',  # good IT
            'inception_v4',  # good i2n
            'vgg-16',  # bad
        ]

        figs = {
            'brain-score': BrainScorePlot(highlighted_models=highlighted_models),
            'brain-score-zoom': BrainScoreZoomPlot(highlighted_models=highlighted_models),
            'individual': IndividualPlots(highlighted_models=highlighted_models),
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
