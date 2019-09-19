import pytest
from pytest import approx

from candidate_models import brain_translated_pool, score_model


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_alexnet():
    model = brain_translated_pool['alexnet']
    score = score_model(model_identifier='alexnet', model=model, benchmark_identifier='dicarlo.Rajalingham2018-i2n')
    assert score.raw.sel(aggregation='center') == approx(.253, abs=.005)
