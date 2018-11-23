import pickle
import pandas as pd

from brainscore.assemblies import DataPoint
from brainscore.metrics import Score

model = 's'

score_v4 = pd.read_pickle(
    f'output/candidate_models._score_model/model_identifier=cornet_{model},'
    f'benchmark_identifier=dicarlo.Majaj2015.V4,weights=imagenet,pca_components=1000,image_size=224.pkl')[
    'data']
score_it = pd.read_pickle(
    f'output/candidate_models._score_model/model_identifier=cornet_{model},'
    f'benchmark_identifier=dicarlo.Majaj2015.IT,weights=imagenet,pca_components=1000,image_size=224.pkl')[
    'data']
score_v4 = score_v4.expand_dims('region')
score_v4['region'] = ['V4']
score_it = score_it.expand_dims('region')
score_it['region'] = ['IT']

score = Score.merge(score_v4, score_it)
score = score.expand_dims('ceiled')
score['ceiled'] = [False]
del score.attrs['raw']

s = DataPoint.__new__(DataPoint)
s.aggregation = score

with open(
        f"output/candidate_models._score_physiology/model=cornet_{model},weights=imagenet,pca_components=1000,"
        f"neural_data=dicarlo.Majaj2015,target_splits=('region',),metric_name=neural_fit.pkl",
        'wb') as f:
    pickle.dump({'data': s}, f)
