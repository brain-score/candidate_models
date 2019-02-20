import pytest

from candidate_models import brain_translated_pool


class TestBestLayers:
    @pytest.mark.parametrize(['model_identifier', 'expected'], [
        ('alexnet', {'V1': 'features.5', 'V2': 'features.5', 'V4': 'features.5', 'IT': 'features.12'}),
        ('CORnet-S', {'V1': 'V1.output-t0', 'V2': 'V2.output-t1', 'V4': 'V4.output-t0', 'IT': 'IT.output-t0'}),
    ])
    def test(self, model_identifier, expected):
        model = brain_translated_pool[model_identifier]
        assert model.layer_model.region_layer_map == expected
