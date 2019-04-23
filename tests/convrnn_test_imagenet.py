import pytest
from brainscore.benchmarks.imagenet import Imagenet2012

from candidate_models import brain_translated_pool

def test_top1(self, model):
    model = brain_translated_pool[model]
    benchmark = Imagenet2012()
    score = benchmark(model)
    accuracy = score.sel(aggregation='center')
    print('Accuracy', accuracy)

if __name__ == '__main__':
    test_top1(model='convrnn_224')
