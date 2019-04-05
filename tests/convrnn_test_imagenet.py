import pytest
from brainscore.benchmarks.imagenet import Imagenet2012

from candidate_models import brain_translated_pool


class TestImagenet:
    @pytest.mark.parametrize(['model', 'expected_top1'], [
        ('resnet-101_v2', .770),
    ])
    def test_top1(self, model, expected_top1):
        model = brain_translated_pool[model]
        benchmark = Imagenet2012()
        score = benchmark(model)
        accuracy = score.sel(aggregation='center')
        print('Accuracy', accuracy)
        assert accuracy == expected_top1

if __name__ == '__main__':
    ff_test = TestImagenet()
#    ff_test.test_top1(model='inception_v1', expected_top1=0.770)
#    ff_test.test_top1(model='resnet-101_v1', expected_top1=0.770)
#    ff_test.test_top1(model='resnet-101_v2', expected_top1=0.770)
    ff_test.test_top1(model='convrnn_224', expected_top1=0.73)

