from candidate_models.models.implementations import model_layers


class TestModelLayers:
    def test_alexnet(self):
        layers = model_layers['alexnet']
        assert layers == \
               ['features.2', 'features.5', 'features.7', 'features.9', 'features.12', 'classifier.2', 'classifier.5']
