from candidate_models.base_models import base_model_pool
from candidate_models.base_models.layers_as_timesteps import LayersAsTimesteps
from candidate_models.model_commitments.ml_pool import model_layers, commitment_assemblies


def resnet_layer_timesteps():
    brain_model = LayersAsTimesteps('resnet-101_v2-layer_timesteps',
                                    activations_model=base_model_pool['resnet-101_v2'],
                                    layers=model_layers['resnet-101_v2'])
    for region, assembly in commitment_assemblies.items():
        brain_model.commit_region(region, assembly)
    return brain_model
