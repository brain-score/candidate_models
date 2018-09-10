import numpy as np
from pytest import approx

from candidate_models import score_model
from candidate_models.models.implementations import model_layers


class TestNewModel:
    def test_pytorch(self):
        import torch
        from torch import nn
        from candidate_models.models.implementations.pytorch import PytorchModel

        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
                self.relu1 = torch.nn.ReLU()
                linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
                self.linear = torch.nn.Linear(int(linear_input_size), 1000)
                self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                x = self.relu2(x)
                return x

        class MyModelWrapper(PytorchModel):
            def __init__(self, *args, **kwargs):
                super().__init__('mymodel', *args, **kwargs)

            def _create_model(self, weights):
                my_model = MyModel()
                assert weights is None  # weight loading would go here
                return my_model

        score = score_model(model=MyModelWrapper, model_identifier='test_pytorch',
                            layers=['linear', 'relu2'], weights=None, pca_components=None)
        score = score.aggregation.sel(aggregation='center')
        assert len(score['layer']) == 2
        np.testing.assert_almost_equal(score.sel(region='V4', layer='relu2'), .281037, decimal=2)


class TestScorePhysiologyMajaj2015:
    def test_alexnet(self):
        score = score_physiology(model='alexnet', layers=model_layers['alexnet'], neural_data='dicarlo.Majaj2015')
        assert score.center.sel(region='IT').max() == approx(0.64, rel=0.005)  # was 0.58
        assert score.center.sel(region='V4', layer='features.2') == approx(0.58, rel=0.005)  # was 0.36
        assert score.center.sel(region='IT', layer='classifier.6') == approx(0.50, rel=0.005)  # was 0.45

    def test_alexnet_conv5(self):
        score = score_physiology(model='alexnet', layers=['features.12'], neural_data='dicarlo.Majaj2015')
        assert score.center.sel(region='IT') == approx(0.62, rel=0.005)
        assert score.center.sel(region='V4') == approx(0.52, rel=0.005)

    def test_alexnet_multilayer(self):
        score = score_physiology(model='alexnet', layers=[['features.12', 'classifier.2']],
                                 neural_data='dicarlo.Majaj2015')
        assert len(np.unique(score.center['layer'])) == 1
        assert np.unique(score.center['layer']) == 'features.12,classifier.2'
        assert score.center.sel(region='IT') == approx(0.63, rel=0.005)
        assert score.center.sel(region='V4') == approx(0.53, rel=0.005)
