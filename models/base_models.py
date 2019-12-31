from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers


def get_model_list():
    return base_model_pool.keys()


def get_model(name):
    return base_model_pool[name]


def get_layers(name):
    return model_layers[name] if name in model_layers else None


if __name__ == '__main__':
    print(get_model_list())
