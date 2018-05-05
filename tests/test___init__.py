from pytest import approx

from neurality import score_physiology


class TestScorePhysiology:
    def test_alexnet_conv5(self):
        score = score_physiology(model='alexnet', layers=['features.12'], neural_data='dicarlo.Majaj2015')
        assert score.center == approx(0.58, rel=0.005)
