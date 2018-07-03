import numpy as np
from pytest import approx

from candidate_models import score_physiology
from candidate_models.models.implementations import model_layers


class TestScorePhysiology:
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
