from model_tools.brain_transformation import ModelCommitment
import warnings
from brainscore.utils import LazyLoad
from brainscore.submission.utils import UniqueKeyDict


class StochasticModelCommitment(ModelCommitment):
    """
    Similar to ModelCommitment but gets model activations multiple times depending on the number of trials. To be
    used with models that have stochastic activations.
    """

    def look_at(self, stimuli, number_of_trials=1):
        stimuli_identifier = stimuli.identifier
        for trial_number in range(number_of_trials):
            stimuli.identifier = stimuli_identifier + '-trial' + '{0:03d}'.format(trial_number)
            if trial_number == 0:
                activations = super().look_at(stimuli, number_of_trials=1)
            else:
                activations += super().look_at(stimuli, number_of_trials=1)
        stimuli.identifier = stimuli_identifier
        return activations/number_of_trials


class StochasticBrainPool(UniqueKeyDict):
    def __init__(self, base_model_pool, model_layers, reload=True):
        super(StochasticBrainPool, self).__init__(reload)
        self.reload = True
        for basemodel_identifier, activations_model in base_model_pool.items():
            if basemodel_identifier not in model_layers:
                warnings.warn(f"{basemodel_identifier} not found in model_layers")
                continue
            model_layer = model_layers[basemodel_identifier]

            def load(identifier=basemodel_identifier, activations_model=activations_model, layers=model_layer):
                assert hasattr(activations_model, 'reload')
                activations_model.reload()
                from candidate_models.model_commitments.stochastic import StochasticModelCommitment
                brain_model = StochasticModelCommitment(identifier=identifier, activations_model=activations_model,
                                                        layers=layers)
                return brain_model

            self[basemodel_identifier] = LazyLoad(load)
