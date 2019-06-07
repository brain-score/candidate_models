import functools
import numpy as np
from pytest import approx
from typing import Union

from brainscore.utils import LazyLoad
from candidate_models import score_model, brain_translated_pool
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import Hooks
from model_tools.activations import PytorchWrapper
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation import LayerMappedModel, TemporalIgnore
from tests.flags import memory_intense


class TestPreselectedLayer:
    def layer_candidate(self, model_name, layer, region, pca_components: Union[None, int] = 1000):
        def load(model_name=model_name, layer=layer, region=region, pca_components=pca_components):
            activations_model = base_model_pool[model_name]
            if pca_components:
                LayerPCA.hook(activations_model, n_components=pca_components)
                activations_model.identifier += Hooks.HOOK_SEPARATOR + "pca_1000"
            model = LayerMappedModel(f"{model_name}-{layer}", activations_model=activations_model)
            model.commit(region, layer)
            model = TemporalIgnore(model)
            return model

        return LazyLoad(load)  # lazy-load to avoid loading all models right away

    @memory_intense
    def test_alexnet_conv2_V4(self):
        model = self.layer_candidate('alexnet', layer='features.5', region='V4', pca_components=1000)
        score = score_model(model_identifier='alexnet-f5-pca_1000', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.V4-pls')
        assert score.raw.sel(aggregation='center').max() == approx(0.656703, abs=0.005)

    @memory_intense
    def test_alexnet_conv5_V4(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='V4', pca_components=1000)
        score = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.V4-pls')
        assert score.raw.sel(aggregation='center') == approx(0.533175, abs=0.005)

    @memory_intense
    def test_alexnet_conv5_IT(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT', pca_components=1000)
        score = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.IT-pls')
        assert score.raw.sel(aggregation='center') == approx(0.601174, abs=0.005)

    @memory_intense
    def test_alexnet_conv3_IT_mask(self):
        model = self.layer_candidate('alexnet', layer='features.6', region='IT', pca_components=None)
        np.random.seed(123)
        score = score_model(model_identifier='alexnet-f6', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.IT-mask')
        assert score.raw.sel(aggregation='center') == approx(0.614621, abs=0.005)

    @memory_intense
    def test_repeat_same_result(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT', pca_components=1000)
        score1 = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                             benchmark_identifier='dicarlo.Majaj2015.IT-pls')
        score2 = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                             benchmark_identifier='dicarlo.Majaj2015.IT-pls')
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
        layer = 'relu2'
        candidate = LayerMappedModel(f"{model_id}-{layer}", activations_model=activations_model)
        candidate.commit('IT', layer)
        candidate = TemporalIgnore(candidate)

        ceiled_score = score_model(model_identifier=model_id, model=candidate,
                                   benchmark_identifier='dicarlo.Majaj2015.IT-pls')
        score = ceiled_score.raw
        assert score.sel(aggregation='center') == approx(.0820823, abs=.005)


class TestPreselectedLayerTemporal:
    def layer_candidate(self, model_name, layer, region, pca_components: Union[None, int] = 1000):
        def load(model_name=model_name, layer=layer, region=region, pca_components=pca_components):
            activations_model = base_model_pool[model_name]
            if pca_components:
                LayerPCA.hook(activations_model, n_components=pca_components)
                activations_model.identifier += Hooks.HOOK_SEPARATOR + "pca_1000"
            model = LayerMappedModel(f"{model_name}-{layer}", activations_model=activations_model)
            model = TemporalIgnore(model)
            model.commit(region, layer)
            return model

        return LazyLoad(load)  # lazy-load to avoid loading all models right away

    @memory_intense
    def test_alexnet_conv5_IT_temporal(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT', pca_components=1000)
        score = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.temporal.IT-pls')
        assert score.raw.sel(aggregation='center') == approx(0.277449, abs=0.005)
        assert len(score.raw.raw['time_bin']) == 12


@memory_intense
class TestBrainTranslated:
    def test_alexnet_pca(self):
        identifier = 'alexnet--pca_1000'
        model = brain_translated_pool[identifier]
        score = score_model(identifier, 'dicarlo.Majaj2015.IT-pls', model=model)
        assert score.raw.sel(aggregation='center') == approx(0.601174, abs=0.005)

    def test_alexnet_degrees(self):
        identifier = 'alexnet--degrees'
        model = brain_translated_pool[identifier]
        score = score_model(identifier, 'dicarlo.Majaj2015.IT-pls', model=model)
        assert score.raw.sel(aggregation='center') == approx(0.560698, abs=0.005)

    def test_alexnet_degrees_pca(self):
        identifier = 'alexnet--degrees-pca_1000'
        model = brain_translated_pool[identifier]
        score = score_model(identifier, 'dicarlo.Majaj2015.IT-pls', model=model)
        assert score.raw.sel(aggregation='center') == approx(0.567, abs=0.005)

    def test_alexnet_temporal_V4(self):
        identifier = 'alexnet--pca_1000'
        model = brain_translated_pool[identifier]
        score = score_model(identifier, 'dicarlo.Majaj2015.temporal.V4-pls', model=model)
        assert score.raw.sel(aggregation='center') == approx(0.281565, abs=0.005)
        assert len(score.raw.raw['time_bin']) == 12

    def test_alexnet_temporal_IT(self):
        identifier = 'alexnet--pca_1000'
        model = brain_translated_pool[identifier]
        score = score_model(identifier, 'dicarlo.Majaj2015.temporal.IT-pls', model=model)
        assert score.raw.sel(aggregation='center') == approx(0.277449, abs=0.005)
        assert len(score.raw.raw['time_bin']) == 12

    def test_alexnet_temporal_V1(self):
        identifier = 'alexnet--pca_1000'
        model = brain_translated_pool[identifier]
        score = score_model(identifier, 'movshon.FreemanZiemba2013.temporal.V1-pls', model=model)
        assert score.raw.sel(aggregation='center') == approx(0.120222, abs=0.005)
        assert len(score.raw.raw['time_bin']) == 25

    def test_alexnet_temporal_V2(self):
        identifier = 'alexnet--pca_1000'
        model = brain_translated_pool[identifier]
        score = score_model(identifier, 'movshon.FreemanZiemba2013.temporal.V2-pls', model=model)
        assert score.raw.sel(aggregation='center') == approx(0.138038, abs=0.005)
        assert len(score.raw.raw['time_bin']) == 25
