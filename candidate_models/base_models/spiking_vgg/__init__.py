import numpy as np
import torch
from torch import nn
from torchvision import transforms

from model_tools.activations.core import ActivationsExtractorHelper
from model_tools.activations.pytorch import load_images
from .model import VGG_SNN_STDB


def create_model(timesteps=500, leak_mem=1.0, scaling_threshold=0.8, reset_threshold=0.0, default_threshold=1.0,
                 activation='STDB', architecture='VGG16', labels=1000, pretrained=True):
    model = VGG_SNN_STDB(vgg_name=architecture, activation=activation, labels=labels, timesteps=timesteps,
                         leak_mem=leak_mem)
    if pretrained:
        pretrained_state = './snn_vgg16_imagenet.pth'
        state = torch.load(pretrained_state, map_location='cpu')
        model.load_state_dict(state['state_dict'])
    model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
    # VGG16 Imagenet thresholds
    ann_thresholds = [10.16, 11.49, 2.65, 2.30, 0.77, 2.75, 1.33, 0.67, 1.13, 1.12, 0.43, 0.73, 1.08, 0.16, 0.58]
    model.module.threshold_init(scaling_threshold=scaling_threshold, reset_threshold=reset_threshold,
                                thresholds=ann_thresholds[:], default_threshold=default_threshold)
    model.eval()
    model.module.network_init(timesteps)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    def load_and_preprocess(image_filepaths):
        images = load_images(image_filepaths)
        images = [transform(image) for image in images]
        return np.concatenate(images)

    def get_activations(preprocessed_inputs, layer_names):
        ...  # TODO

    model = ActivationsExtractorHelper(identifier='spiking-vgg',
                                       get_activations=get_activations, preprocessing=load_and_preprocess,
                                       batch_size=35)
    return model
