import argparse
import logging
import sys

from candidate_models import model_activations, model_layers
from candidate_models.models import models, Defaults, infer_image_size
from candidate_models.models.implementations import Defaults as DeepModelDefaults

_logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser('model comparison')
    parser.add_argument('--model', type=str, required=True, choices=list(models.keys()))
    parser.add_argument('--weights', type=str, default=DeepModelDefaults.weights)
    parser.add_argument('--no-model_weights', action='store_const', const=None, dest='model_weights')
    parser.add_argument('--layers', nargs='+', default=None)
    parser.add_argument('--pca', type=int, default=DeepModelDefaults.pca_components,
                        help='Number of components to reduce the flattened features to')
    parser.add_argument('--no-pca', action='store_const', const=None, dest='pca')
    parser.add_argument('--image_size', type=int, default=-1,
                        help='size of the image (same value is used for width and height). '
                             '-1 (default) to infer from model')
    parser.add_argument('--stimulus_set', type=str, default=Defaults.stimulus_set)
    parser.add_argument('--batch_size', type=int, default=DeepModelDefaults.batch_size)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    args.layers = args.layers or model_layers[args.model]
    if args.image_size == -1:
        args.image_size = infer_image_size(args.model)
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3']:
        logging.getLogger(disable_logger).setLevel(logging.WARNING)
    _logger.info("Running with args %s", vars(args))

    model_activations(model=args.model, layers=args.layers, image_size=args.image_size,
                      stimulus_set=args.stimulus_set, pca_components=args.pca, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
