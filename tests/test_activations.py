from brainscore import get_stimulus_set
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers
from model_tools.activations.pca import LayerPCA


class TestActivations:
    def test_alexnet(self):
        activations_model = base_model_pool['alexnet']
        LayerPCA.hook(activations_model, n_components=1000)
        stimulus_set = get_stimulus_set('dicarlo.hvm')
        stimulus_set.name = 'dicarlo.hvm'
        activations = activations_model.from_stimulus_set(stimulus_set, layers=model_layers['alexnet'])
        assert activations is not None
