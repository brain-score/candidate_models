import numpy as np
from pytest import approx

from candidate_models import score_model


class TestBrainScore:
    def test_alexnet(self):
        score = score_model(model='alexnet')
        assert score == approx(0.610, rel=0.005)

    def test_raw_dimensions(self):
        score = score_model(model='alexnet')
        raw_values = score.attrs['raw']
        assert {'benchmark', 'layer', 'aggregation'} == set(raw_values.dims)
        assert len(raw_values['layer']) == 7
        assert len(raw_values['benchmark']) == 2
        assert len(raw_values['aggregation']) == 2

    def test_raw_aggregate(self):
        score = score_model(model='alexnet')
        raw_values = score.attrs['raw']
        assert raw_values.sel(aggregation='center', benchmark='dicarlo.Majaj2015.IT', layer='features.12') == \
               approx(.589, abs=0.005)

    def test_raw_raw(self):
        score = score_model(model='alexnet')
        benchmark_values = score.attrs['raw']
        raw_values = benchmark_values.attrs['raw']
        assert {'benchmark', 'layer', 'split', 'neuroid'} == set(raw_values.dims)
        assert len(raw_values['benchmark']) == 2


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
        raw_scores = score.attrs['raw']
        assert raw_scores.sel(benchmark='dicarlo.Majaj2015.V4').max() == approx(.305, abs=.005)
        assert raw_scores.sel(benchmark='dicarlo.Majaj2015.IT').max() == approx(.189, abs=.005)


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
