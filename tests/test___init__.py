import sys

import logging
from pytest import approx

from neurality import score_physiology
from neurality.models.implementations import model_layers

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')


class TestScorePhysiology:
    def test_alexnet(self):
        score = score_physiology(model='alexnet', layers=model_layers['alexnet'], neural_data='dicarlo.Majaj2015')
        assert score.center == approx(0.58, rel=0.005)

    def test_alexnet_conv5(self):
        score = score_physiology(model='alexnet', layers=['features.12'], neural_data='dicarlo.Majaj2015')
        assert score.center.sel(region='IT') == approx(0.62, rel=0.005)
        assert score.center.sel(region='V4') == approx(0.52, rel=0.005)
