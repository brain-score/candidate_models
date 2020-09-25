from model_tools.brain_transformation import ModelCommitment


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
