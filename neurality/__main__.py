import argparse
import logging
import os
import sys

from neurality import models, score_model, Defaults
from neurality.plot import plot_scores, results_dir
from neurality.storage import get_function_identifier

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_weights', type=str, default=models.Defaults.model_weights)
    parser.add_argument('--layers', type=str, nargs='+', required=True)
    parser.add_argument('--neural_data', type=str, default=Defaults.neural_data)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    logger.info("Running with args %s", vars(args))

    logger.info('Scoring model')
    scores = score_model(args.model, args.layers, neural_data=args.neural_data, model_weights=args.model_weights)

    logger.info('Plotting')
    function_identifier = get_function_identifier(score_model, vars(args))
    output_filepath = os.path.join(results_dir, function_identifier + '.svg')
    plot_scores(scores, output_filepath=output_filepath)
