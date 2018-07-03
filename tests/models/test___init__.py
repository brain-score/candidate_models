import numpy as np

from candidate_models.models import model_activations
from candidate_models.models import model_multi_activations
from candidate_models.models.implementations import model_layers
from tests.models import patch_imagenet


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


class TestModelActivations:
    def test_single_layer(self, mocker):
        patch_imagenet(mocker)
        activations = model_activations(model='alexnet', layers=['features.12'])
        assert unique_preserved_order(activations['layer']) == 'features.12'

    def test_two_layers(self, mocker):
        patch_imagenet(mocker)
        activations = model_activations(model='alexnet', layers=['features.12', 'classifier.2'])
        np.testing.assert_array_equal(unique_preserved_order(activations['layer']), ['features.12', 'classifier.2'])

    def test_all_layers(self, mocker):
        patch_imagenet(mocker)
        layers = model_layers['alexnet']
        activations = model_activations(model='alexnet', layers=layers)
        np.testing.assert_array_equal(unique_preserved_order(activations['layer']), layers)


class TestModelMultiActivations:
    def test_single_layer(self, mocker):
        patch_imagenet(mocker)
        activations = model_multi_activations(model='alexnet', multi_layers=[['features.12']])
        assert unique_preserved_order(activations['layer']) == 'features.12'

    def test_combine_two(self, mocker):
        patch_imagenet(mocker)
        activations = model_multi_activations(model='alexnet', multi_layers=[['features.12', 'classifier.2']])
        assert len(np.unique(activations['layer'])) == 1
        assert unique_preserved_order(activations['layer']) == 'features.12,classifier.2'

    def test_combine_two_two(self, mocker):
        patch_imagenet(mocker)
        activations = model_multi_activations(model='alexnet',
                                              multi_layers=[['features.2', 'features.5'],
                                                            ['features.12', 'classifier.2']])
        assert len(np.unique(activations['layer'])) == 2
        np.testing.assert_array_equal(unique_preserved_order(activations['layer']),
                                      ['features.2,features.5', 'features.12,classifier.2'])
