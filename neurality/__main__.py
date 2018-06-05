import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from neurality import score_model, Defaults
from neurality.models import models
from neurality.models.implementations import Defaults as DeepModelDefaults
from neurality.models.implementations import model_layers

logger = logging.getLogger(__name__)

model_meta = pd.read_csv(os.path.join(os.path.dirname(__file__), 'models', 'models.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=models.keys())
    parser.add_argument('--model_weights', type=str, default=DeepModelDefaults.weights)
    parser.add_argument('--layers', type=str, nargs='+', default=None)
    parser.add_argument('--pca', type=int, default=DeepModelDefaults.pca_components,
                        help='Number of components to reduce the flattened features to')
    parser.add_argument('--no-pca', action='store_const', const=None, dest='pca')
    parser.add_argument('--neural_data', type=str, default=Defaults.neural_data)
    parser.add_argument('--image_size', type=int, default=DeepModelDefaults.image_size)
    parser.add_argument('--metric', type=str, default=Defaults.metric_name)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    print(args.model)
    meta = model_meta[model_meta['model'] == args.model]
    image_size = meta['image_size']
    if len(image_size) == 1 and not np.isnan(image_size.values[0]):
        args.image_size = int(image_size.values[0])
    else:
        logger.warning("Could not lookup image size")
    args.layers = args.layers or (model_layers[args.model] if not args.model.startswith('basenet')
                                  else ['basenet-layer_v4', 'basenet-layer_pit', 'basenet-layer_ait'])
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    logger.info("Running with args %s", vars(args))

    logger.info('Scoring model')
    score = score_model(args.model, layers=args.layers, weights=args.model_weights,
                        pca_components=args.pca, image_size=args.image_size,
                        neural_data=args.neural_data, metric_name=args.metric)
    print(score.center)


main()
