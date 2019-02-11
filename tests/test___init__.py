import functools
import numpy as np
from pytest import approx

from candidate_models import score_model, brain_translated_pool
from candidate_models.base_models import base_model_pool
from model_tools.activations import PytorchWrapper
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation import LayerModel


class TestPreselectedLayer:
    def layer_candidate(self, model_name, layer, region, pca_components=1000):
        activations_model = base_model_pool[model_name]
        if pca_components:
            LayerPCA.hook(activations_model, n_components=pca_components)

        return LayerModel(model_name, base_model=activations_model, layer=layer, region=region)

    def test_alexnet_conv2_V4(self):
        model = self.layer_candidate('alexnet', layer='features.5', region='V4')
        score = score_model(model_identifier='alexnet-f5', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.V4-regressing')
        assert score.raw.sel(aggregation='center').max() == approx(0.6508, rel=0.005)

    def test_alexnet_conv5_V4(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='V4')
        score = score_model(model_identifier='alexnet-f12', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.V4-regressing')
        assert score.raw.sel(aggregation='center') == approx(0.530405, rel=0.005)

    def test_alexnet_conv5_IT(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT')
        score = score_model(model_identifier='alexnet-f12', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.IT-regressing')
        assert score.raw.sel(aggregation='center') == approx(0.601174, rel=0.005)

    def test_repeat_same_result(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT')
        score1 = score_model(model_identifier='alexnet-f12', model=model,
                             benchmark_identifier='dicarlo.Majaj2015.IT-regressing')
        score2 = score_model(model_identifier='alexnet-f12', model=model,
                             benchmark_identifier='dicarlo.Majaj2015.IT-regressing')
        assert (score1 == score2).all()

    def test_newmodel_pytorch(self):
        import torch
        from torch import nn
        from model_tools.activations.pytorch import load_preprocess_images

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

        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        model_id = 'new_pytorch'
        activations_model = PytorchWrapper(model=MyModel(), preprocessing=preprocessing, identifier=model_id)
        candidate = LayerModel(model_id, base_model=activations_model, layer='relu2', region='IT')

        ceiled_score = score_model(model_identifier=model_id, model=candidate,
                                   benchmark_identifier='dicarlo.Majaj2015.IT-regressing')
        score = ceiled_score.raw
        assert score.sel(aggregation='center') == approx(.078191, abs=.001)


class TestBrainTranslated:
    def test_alexnet(self):
        model = brain_translated_pool['alexnet']
        score = score_model('alexnet', 'dicarlo.Majaj2015.IT-regressing', model=model)
        assert score.raw.sel(aggregation='center') == approx(0.601174, rel=0.005)
