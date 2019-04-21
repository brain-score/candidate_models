from pytest import approx

from candidate_models import brain_translated_pool, score_model

model_identifier = 'resnet-101_v2'
model = brain_translated_pool[model_identifier]
score = score_model(model_identifier=model_identifier, model=model, benchmark_identifier='dicarlo.Rajalingham2018-i2n')
print('Computed SCORE', score.raw.sel(aggregation='center'))
#assert score.raw.sel(aggregation='center') == approx(.253, abs=.005)
