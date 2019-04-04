import numpy as np
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers
stimuli_path = '/home/anayebi/candidate_models/examples/image.jpg'

model = base_model_pool['convrnn_224']
layers = model_layers['convrnn_224']
layers = [layers[-1]]
activations = model(stimuli=[stimuli_path], layers=layers)
