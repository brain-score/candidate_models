import numpy as np

from neurality import model_activations, model_layers


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


class TestModelActivations:
    def test_single_layer(self):
        activations = model_activations(model='alexnet', layers=['features.12'])
        assert unique_preserved_order(activations['layer']) == 'features.12'

    def test_two_layers(self):
        activations = model_activations(model='alexnet', layers=['features.12', 'classifier.2'])
        np.testing.assert_array_equal(unique_preserved_order(activations['layer']), ['features.12', 'classifier.2'])

    def test_all_layers(self):
        layers = model_layers['alexnet']
        activations = model_activations(model='alexnet', layers=layers)
        np.testing.assert_array_equal(unique_preserved_order(activations['layer']), layers)
