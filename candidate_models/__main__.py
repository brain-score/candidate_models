import logging
import sys

import argparse

from candidate_models import score_model, map_and_score_model, mapping_model_pool
from candidate_models.base_models import base_model_pool

logger = logging.getLogger(__name__)


def main():
    score_functions = {'score': score_model, 'map_and_score': map_and_score_model}
    score_pools = {'score': base_model_pool, 'map_and_score': mapping_model_pool}

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
    score_pool = score_pools[args.scoring_type]
    model = score_pool[args.model]
    score = score_function(args.model, model=model, benchmark_identifier=args.benchmark)
    print(score)


main()
