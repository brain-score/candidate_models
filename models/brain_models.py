from candidate_models.model_commitments import cornet_brain_pool
from brainscore.submission import test_models

"""
Template module for a brain model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return list(cornet_brain_pool.keys())


def get_model(name):
    """
    This method fetches an instance of a brain model. The instance has to implement the BrainModel interface in the
    brain-score project(see imports). To get a detailed explanation of how the interface hast to be implemented,
    check out the brain-score project(https://github.com/brain-score/brain-score), examples section :param name: the
    name of the model to fetch
    :return: the model instance, which implements the BrainModel interface
    """
    return cornet_brain_pool[name]


if __name__ == '__main__':
    # Use this method to ensure the correctness of the brain model implementations.
    # It executes a mock run of brain-score benchmarks.
    test_models.check_brain_models(__name__)
