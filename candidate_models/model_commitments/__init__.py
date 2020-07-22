from candidate_models.base_models import base_model_pool
from candidate_models.base_models import BaseModelPool

from candidate_models.model_commitments.cornets import cornet_brain_pool
from candidate_models.model_commitments.model_layer_def import model_layers
from candidate_models.model_commitments.vs_layer import visual_search_layer

from brainscore.submission.ml_pool import MLBrainPool
from brainscore.submission.utils import UniqueKeyDict

brain_translated_pool = UniqueKeyDict(reload=True)

ml_brain_pool = MLBrainPool(base_model_pool, model_layers)

for identifier, model in ml_brain_pool.items():
    brain_translated_pool[identifier] = model

for identifier, model in cornet_brain_pool.items():
    brain_translated_pool[identifier] = model

def MLSearchPool(target_img_size=28, search_image_size=224):
    target_model_pool = BaseModelPool(input_size=target_img_size)
    stimuli_model_pool = BaseModelPool(input_size=search_image_size)

    vs_model_param = {}
    vs_model_param['tar_pool'] = target_model_pool
    vs_model_param['stim_pool'] = stimuli_model_pool
    vs_model_param['model_layers'] = visual_search_layer
    vs_model_param['tar_size'] = target_img_size
    vs_model_param['stim_size'] = search_image_size

    ml_search_pool = MLBrainPool(base_model_pool, model_layers, vs_model_param=vs_model_param)

    return ml_search_pool
