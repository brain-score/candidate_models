import logging
import sys

import argparse
import fire

from candidate_models import score_layers as score_layers_function, score_model as score_model_function, get_activations
from candidate_models.model_commitments import model_layers_pool, brain_translated_pool

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
args, _ = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


def activations(model, assembly, layers=None):
    if isinstance(layers, str):
        layers = [layers]
    model_layers = model_layers_pool[model]
    model, default_layers = model_layers['model'], model_layers['layers']
    layers = layers or default_layers
    result = get_activations(model, layers=layers, assembly_identifier=assembly)
    print(result)


def score_model(model, benchmark):
    model = brain_translated_pool[model]
    result = score_model_function(model_identifier=model, benchmark_identifier=benchmark, model=model)
    print(result)


def score_layers(model, benchmark, layers=None):
    if isinstance(layers, str):
        layers = [layers]
    model_layers = model_layers_pool[model]
    model, default_layers = model_layers['model'], model_layers['layers']
    layers = layers or default_layers
    result = score_layers_function(model_identifier=model, benchmark_identifier=benchmark, model=model,
                                   layers=layers)
    print(result)


fire.Fire()
