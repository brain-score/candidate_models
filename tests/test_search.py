import pytest
from pytest import approx

from candidate_models.model_commitments import brain_translated_pool
from brainscore.benchmarks import benchmark_pool
import numpy as np
import matplotlib.pyplot as plt
import brainscore

def test_search():
    model = brain_translated_pool['vgg-16']
    score = score_model(model_identifier='vgg-16', model=model, benchmark_identifier='klab.Zhang2018-ObjArray')
    assert score.raw.sel(aggregation='center') == approx(0.407328, abs=.005)
