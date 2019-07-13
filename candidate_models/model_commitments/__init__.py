from candidate_models.model_commitments.cornets import cornet_brain_pool
from candidate_models.model_commitments.ml_pool import ml_brain_pool
from candidate_models.utils import UniqueKeyDict

brain_translated_pool = UniqueKeyDict()

for identifier, model in ml_brain_pool.items():
    brain_translated_pool[identifier] = model

for identifier, model in cornet_brain_pool.items():
    brain_translated_pool[identifier] = model
