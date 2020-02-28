from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers
from model_tools import test_models


def get_model_list():
    return list(base_model_pool.keys())


def get_model(name):
    return base_model_pool[name]


def get_layers(name):
    return model_layers[name] if name in model_layers else None


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    test_models.check_base_models(__name__)
