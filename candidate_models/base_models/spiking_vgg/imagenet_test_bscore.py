#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import pdb
from spiking_model_bscore import *
import sys
import os


use_cuda = True

torch.manual_seed(2)
if torch.cuda.is_available() and use_cuda:
    print ("\n \t ------- Running on GPU -------")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def test(loader):

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        is_best = False
        print_accuracy_every_batch = True
        global max_correct
        
        model.module.network_init(timesteps)
        
        for batch_idx, (data, target) in enumerate(loader):
                        
            if torch.cuda.is_available() and use_cuda:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            output = output/(timesteps)
            
            loss = F.cross_entropy(output,target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            if print_accuracy_every_batch:
                
                print('\nAccuracy: {}/{}({:.2f}%)'.format(
                    correct.item(),
                    (batch_idx+1)*data.size(0),
                    100. * correct.item() / ((batch_idx+1)*data.size(0))
                    )
                )
            
        print('\nTest set: Loss: {:.6f}, Current: {:.2f}%, Best: {:.2f}%\n'.  format(
            total_loss/(batch_idx+1), 
            100. * correct.item() / len(test_loader.dataset),
            100. * max_correct.item() / len(test_loader.dataset)
            )
        )

    
dataset             = 'IMAGENET' 
batch_size          = 35
timesteps           = 500
num_workers         = 4
leak_mem            = 1.0
scaling_threshold   = 0.8
reset_threshold     = 0.0
default_threshold   = 1.0
activation          = 'STDB'
architecture        = 'VGG16'
pretrained          = True
pretrained_state    = './snn_vgg16_imagenet.pth'

if dataset == 'IMAGENET':
    labels      = 1000
    traindir    = os.path.join('/data2/backup/imagenet2012/', 'train')
    valdir      = os.path.join('/data2/backup/imagenet2012/', 'val')
    normalize   = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    trainset    = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
    testset     = datasets.ImageFolder(
                        valdir,
                        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ])) 

train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

model = VGG_SNN_STDB(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak_mem=leak_mem)

if pretrained:
    state = torch.load(pretrained_state, map_location='cpu')
    model.load_state_dict(state['state_dict'])
  
model = nn.DataParallel(model)

if torch.cuda.is_available() and use_cuda:
    model.cuda()


criterion = nn.CrossEntropyLoss()

print('Dataset                  :{} '.format(dataset))
print('Batch Size               :{} '.format(batch_size))
print('Timesteps                :{} '.format(timesteps))

print('Membrane Leak            :{} '.format(leak_mem))
print('Scaling Threshold        :{} '.format(scaling_threshold))
print('Activation               :{} '.format(activation))
print('Architecture             :{} '.format(architecture))
if pretrained:
    print('Pretrained Weight File   :{} '.format(pretrained_state))

print('Criterion                :{} '.format(criterion))
print('\n{}'.format(model))

start_time = datetime.datetime.now()

#VGG16 Imagenet thresholds
ann_thresholds = [10.16, 11.49, 2.65, 2.30, 0.77, 2.75, 1.33, 0.67, 1.13, 1.12, 0.43, 0.73, 1.08, 0.16, 0.58]

model.module.threshold_init(scaling_threshold=scaling_threshold, reset_threshold=reset_threshold, thresholds = ann_thresholds[:], default_threshold=default_threshold)

 
test(test_loader)    
