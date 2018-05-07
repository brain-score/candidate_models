import logging
import sys

import seaborn
from matplotlib import pyplot

from mkgu.assemblies import merge_data_arrays
from neurality import models, score_physiology

seaborn.set()


def plot_model_pcas(model, layer, model_weights=models.Defaults.model_weights, stimulus_set='dicarlo.Majaj2015',
                    pcas=(50, 100, 200, 500, 1000, 1500, 2000)):
    centers, errs = [], []
    for pca in pcas:
        print("PCA {}".format(pca))
        score = score_physiology(model=model, model_weights=model_weights, layers=[layer], pca_components=pca,
                                 neural_data=stimulus_set)
        center, err = score.center, score.error
        center, err = center.expand_dims('pca'), err.expand_dims('pca')
        center['pca'] = [pca]
        err['pca'] = [pca]
        centers.append(center)
        errs.append(err)
    centers, errs = merge_data_arrays(centers), merge_data_arrays(errs)
    centers, errs = centers.sel(region='IT').squeeze('layer'), errs.sel(region='IT').squeeze('layer')

    pyplot.errorbar(x=centers['pca'], y=centers.values, yerr=errs.values)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plot_model_pcas('alexnet', 'features.12')
    pyplot.savefig('results/plot.jpg')
