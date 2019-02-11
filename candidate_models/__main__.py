import logging
import sys

import argparse

from candidate_models import score_layers, score_model
from candidate_models.model_commitments import model_layers_pool, brain_translated_pool

logger = logging.getLogger(__name__)


def main():
    score_functions = {'score_layers': score_layers, 'score': score_model}
    model_pools = {'score_layers': model_layers_pool, 'score': brain_translated_pool}

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--scoring_type', type=str, choices=score_functions.keys(), default='score')
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3']:
        logging.getLogger(disable_logger).setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logger.info("Running with args %s", vars(args))

    logger.info('Scoring model')
    score_function = score_functions[args.scoring_type]
    score_pool = model_pools[args.scoring_type]
    model = score_pool[args.model]
    if not isinstance(model, dict):
        score = score_function(args.model, model=model, benchmark_identifier=args.benchmark)
    else:
        score = score_function(args.model, **model, benchmark_identifier=args.benchmark)
    print(score)


main()
