import warnings

import itertools

from candidate_models.utils import UniqueKeyDict

class VisualSearchLayers(UniqueKeyDict):
    def __init__(self):
        super(VisualSearchLayers, self).__init__()
        layers = {
            'vgg-16': [f'block{i + 1}_pool' for i in range(3,5)],
        }
        for basemodel_identifier, default_layers in layers.items():
            self[basemodel_identifier] = default_layers

    @staticmethod
    def _item(item):
        return item

    def __getitem__(self, item):
        return super(VisualSearchLayers, self).__getitem__(self._item(item))

    def __contains__(self, item):
        return super(VisualSearchLayers, self).__contains__(self._item(item))

visual_search_layer = VisualSearchLayers()
