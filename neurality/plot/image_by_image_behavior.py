import logging
import os
import sys

import numpy as np
from matplotlib import pyplot

from neurality.plot import score_color_mapping, clean_axis


def plot(model, marker_size=2, scatter_alpha=0.7):
    data = np.load(os.path.join(os.path.dirname(__file__), '{}.npy'.format(model)))
    model_responses, human_responses = data[:, 0], data[:, 1]
    fig, ax = pyplot.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([-.5, 5.5])
    ax.set_ylim([-.5, 5.5])
    ticks = [0, 1, 2, 3, 4, 5]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.scatter(model_responses, human_responses, color=score_color_mapping['behavior'], alpha=scatter_alpha,
               s=marker_size)
    # diagonal
    minmax = [model_responses.min(), model_responses.max()]
    ax.plot(minmax, minmax, color='gray', linestyle='dashed', linewidth=1.)
    # label
    ax.set_xlabel('model')
    ax.set_ylabel('human')
    clean_axis(ax)
    pyplot.tight_layout()
    return fig


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fit = None

    models = ['pnasnet_large']
    figs = {model: plot(model) for model in models}
    for model, fig in figs.items():
        savepath = os.path.join(os.path.dirname(__file__), '..', '..', 'results/image-by-image/{}.svg'.format(model))
        fig.savefig(savepath, format='svg')
