import functools
import numpy as np
import pytest
from pytest import approx
from typing import Union

from brainscore.utils import LazyLoad
from candidate_models import score_model, brain_translated_pool
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments.ml_pool import Hooks
from model_tools.activations import PytorchWrapper
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation import LayerMappedModel, TemporalIgnore


@pytest.mark.private_access
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

    @pytest.mark.memory_intense
    def test_alexnet_conv2_V4(self):
        model = self.layer_candidate('alexnet', layer='features.5', region='V4', pca_components=1000)
        score = score_model(model_identifier='alexnet-f5-pca_1000', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.V4-pls')
        assert score.raw.sel(aggregation='center').max() == approx(0.656703, abs=0.005)

    @pytest.mark.memory_intense
    def test_alexnet_conv5_V4(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='V4', pca_components=1000)
        score = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.V4-pls')
        assert score.raw.sel(aggregation='center') == approx(0.533175, abs=0.005)

    @pytest.mark.memory_intense
    def test_alexnet_conv5_IT(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT', pca_components=1000)
        score = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.IT-pls')
        assert score.raw.sel(aggregation='center') == approx(0.601174, abs=0.005)

    @pytest.mark.memory_intense
    def test_alexnet_conv3_IT_mask(self):
        model = self.layer_candidate('alexnet', layer='features.6', region='IT', pca_components=None)
        np.random.seed(123)
        score = score_model(model_identifier='alexnet-f6', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.IT-mask')
        assert score.raw.sel(aggregation='center') == approx(0.614621, abs=0.005)

    @pytest.mark.memory_intense
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
        assert score.sel(aggregation='center') == approx(.0820823, abs=.01)


@pytest.mark.private_access
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

    @pytest.mark.memory_intense
    def test_alexnet_conv5_IT_temporal(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT', pca_components=1000)
        score = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                            benchmark_identifier='dicarlo.Majaj2015.temporal.IT-pls')
        assert score.raw.sel(aggregation='center') == approx(0.277449, abs=0.005)
        assert len(score.raw.raw['time_bin']) == 12


@pytest.mark.private_access
@pytest.mark.memory_intense
class TestBrainTranslated:
    @pytest.mark.parametrize(['model_identifier', 'expected_score'], [
        ('alexnet--pca_1000', .601174),
        ('alexnet--degrees', .560698),
        ('alexnet--degrees-pca_1000', .567),
        ('CORnet-S', .600),
    ])
    def test_Majaj2015ITpls(self, model_identifier, expected_score):
        model = brain_translated_pool[model_identifier]
        score = score_model(model_identifier, 'dicarlo.Majaj2015.IT-pls', model=model)
        assert score.raw.sel(aggregation='center') == approx(expected_score, abs=0.005)

    @pytest.mark.parametrize(['identifier_suffix', 'benchmark', 'expected_score', 'time_bins'], [
        ('--pca_1000', 'dicarlo.Majaj2015.temporal.V4-pls', .281565, 12),
        ('--pca_1000', 'dicarlo.Majaj2015.temporal.IT-pls', .277449, 12),
        ('--pca_1000', 'movshon.FreemanZiemba2013.temporal.V1-pls', .120222, 25),
        ('--pca_1000', 'movshon.FreemanZiemba2013.temporal.V2-pls', .138038, 25),
    ])
    def test_alexnet_temporal(self, identifier_suffix, benchmark, expected_score, time_bins):
        identifier = f'alexnet{identifier_suffix}'
        model = brain_translated_pool[identifier]
        score = score_model(identifier, benchmark, model=model)
        assert score.raw.sel(aggregation='center') == approx(expected_score, abs=0.005)
        assert len(score.raw.raw['time_bin']) == time_bins

    @pytest.mark.parametrize(['model_identifier', 'expected_score'], [
        ('CORnet-S', .25),
        ('CORnet-R2', .298),
        ('alexnet', np.nan),
    ])
    def test_candidate_Kar2019OST(self, model_identifier, expected_score):
        model = brain_translated_pool[model_identifier]
        score = score_model(model_identifier=model_identifier, model=model, benchmark_identifier='dicarlo.Kar2019-ost')
        if not np.isnan(expected_score):
            assert score.raw.sel(aggregation='center') == approx(expected_score, abs=.002)
        else:
            assert np.isnan(score.raw.sel(aggregation='center'))

    @pytest.mark.parametrize(['model_identifier', 'expected_score'], [
        ('CORnet-S', .382),
        ('alexnet', .253),
    ])
    def test_Rajalingham2018i2n(self, model_identifier, expected_score):
        model = brain_translated_pool[model_identifier]
        score = score_model(model_identifier=model_identifier, model=model,
                            benchmark_identifier='dicarlo.Rajalingham2018-i2n')
        assert score.raw.sel(aggregation='center') == approx(expected_score, abs=.005)
