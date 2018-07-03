import glob
import os
import re

import numpy as np
import seaborn
from matplotlib import pyplot

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
                                                      'output', 'neurality.score_physiology', '*'))]
    models = [re.match('.*/model=(.*),model_weights.*', file) for file in models]
    models = [match.group(1) for match in models if match]
    models = np.unique(models)
    potentially_broken_models = ['resnet-50_v1', 'resnet-101_v1', 'resnet-152_v1']
    models = [model for model in models if model not in potentially_broken_models]
    return models


def clean_axis(ax):
    ax.grid(b=True, which='major', linewidth=0.5)
    seaborn.despine(right=True)
