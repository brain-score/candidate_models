import logging
from collections import OrderedDict
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    KERAS = 1
    PYTORCH = 2
    SLIM = 3


def get_model_type(model):
    import keras
    import torch
    from neurality.models.implementations import TFSlimModel

    if isinstance(model, keras.engine.topology.Container):
        return ModelType.KERAS
    elif isinstance(model, torch.nn.Module):
        return ModelType.PYTORCH
    elif isinstance(model, TFSlimModel):
        return ModelType.SLIM
    else:
        raise ValueError("Unsupported model framework: %s" % str(model))


PYTORCH_SUBMODULE_SEPARATOR = '.'
