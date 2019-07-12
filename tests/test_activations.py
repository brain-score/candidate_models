import pytest

from brainscore import get_stimulus_set
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments.ml_pool import model_layers
from model_tools.activations.pca import LayerPCA
from tests.flags import memory_intense


@memory_intense
class TestActivations:
    @pytest.mark.parametrize('model_identifier', ['alexnet', 'resnet-101_v2'])
    def test_model(self, model_identifier):
        activations_model = base_model_pool[model_identifier]
        LayerPCA.hook(activations_model, n_components=1000)
        stimulus_set = get_stimulus_set('dicarlo.hvm')
        stimulus_set.name = 'dicarlo.hvm'
        activations = activations_model.from_stimulus_set(stimulus_set, layers=model_layers[model_identifier])
        assert activations is not None
