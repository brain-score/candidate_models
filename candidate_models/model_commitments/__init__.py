from brainscore.utils import LazyLoad
from candidate_models.model_commitments.cornets import cornet_brain_pool
from candidate_models.model_commitments.ml_extended import resnet_layer_timesteps
from candidate_models.model_commitments.ml_pool import ml_brain_pool
from candidate_models.utils import UniqueKeyDict

brain_translated_pool = UniqueKeyDict()


def register_brain_model(identifier, model):
    brain_translated_pool[identifier] = model


for identifier, model in ml_brain_pool.items():
    register_brain_model(identifier, model)
register_brain_model('resnet-101_v2-layer_timesteps', LazyLoad(resnet_layer_timesteps))

for identifier, model in cornet_brain_pool.items():
    register_brain_model(identifier, model)
