import numpy as np
from pytest import approx

from candidate_models import score_model


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

                # init weights for reproducibility
                self.conv1.weight.data.fill_(0.01)
                self.conv1.bias.data.fill_(0.01)
                self.linear.weight.data.fill_(0.01)
                self.linear.bias.data.fill_(0.01)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                x = self.relu2(x)
                return x

        class MyModelWrapper(PytorchModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def _create_model(self, weights):
                my_model = MyModel()
                assert weights is None  # weight loading would go here
                return my_model

        score = score_model(model=MyModelWrapper, model_identifier='test_pytorch',
                            benchmark='brain-score',
                            layers=['linear', 'relu2'], weights=None, pca_components=None)
        assert score.attrs['raw'].sel(aggregation='center', benchmark='dicarlo.Majaj2015.V4') == approx(.305, abs=.005)
        assert score.attrs['raw'].sel(aggregation='center', benchmark='dicarlo.Majaj2015.IT') == approx(.189, abs=.005)


class TestMajaj2015:
    def test_alexnet_V4(self):
        score = score_model(model='alexnet', benchmark='dicarlo.Majaj2015.V4')
        assert score.sel(aggregation='center').max() == approx(0.631, rel=0.005)

    def test_alexnet_IT(self):
        score = score_model(model='alexnet', benchmark='dicarlo.Majaj2015.IT')
        assert score.sel(aggregation='center').max() == approx(0.589, rel=0.005)

    def test_same_result(self):
        score1 = score_model(model='alexnet', benchmark='dicarlo.Majaj2015.IT')
        score2 = score_model(model='alexnet', benchmark='dicarlo.Majaj2015.IT')
        assert (score1 == score2).all()

    def test_alexnet_conv5_V4(self):
        score = score_model(model='alexnet', layers=['features.12'], benchmark='dicarlo.Majaj2015.V4')
        assert score.sel(aggregation='center') == approx(0.486, rel=0.005)

    def test_alexnet_conv5_IT(self):
        score = score_model(model='alexnet', layers=['features.12'], benchmark='dicarlo.Majaj2015.IT')
        assert score.sel(aggregation='center') == approx(0.589, rel=0.005)
