import itertools

from brainscore.submission.ml_pool import ModelLayers

layers = {
    'vgg-16': [f'block{i + 1}_pool' for i in range(3,5)],
}

visual_search_layer = ModelLayers(layers)
