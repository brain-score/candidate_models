import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    KERAS = 1
    PYTORCH = 2


def get_model_type(model):
    import keras
    import torch

    if isinstance(model, keras.engine.topology.Container):
        return ModelType.KERAS
    elif isinstance(model, torch.nn.Module):
        return ModelType.PYTORCH
    else:
        raise ValueError("Unsupported model framework: %s" % str(model))


PYTORCH_SUBMODULE_SEPARATOR = '.'
