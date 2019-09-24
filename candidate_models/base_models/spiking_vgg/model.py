# ---------------------------------------------------
# Imports
# ---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy

torch.manual_seed(2)

cfg = {
    'VGG5': [64, 'A', 128, 'D', 128, 'A'],
    'VGG9': [64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'A', 512, 'D', 512, 'D'],
    'VGG11': [64, 'A', 128, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512, 'D', 512, 'D'],
    'VGG16': [64, 'D', 64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512,
              'D', 512, 'D', 512, 'A']
}


class PoissonGenerator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input) * 1.0).float(), torch.sign(input))
        return out


class STDPSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, last_spike):
        ctx.save_for_backward(last_spike)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out


class VGG_SNN_STDB(nn.Module):

    def __init__(self, vgg_name, activation='STDB', labels=1000, timesteps=75, leak_mem=0.99, drop=0.2):
        super().__init__()

        self.timesteps = timesteps
        self.vgg_name = vgg_name
        self.labels = labels
        self.leak_mem = leak_mem
        if activation == 'Linear':
            self.act_func = LinearSpike.apply
        elif activation == 'STDB':
            self.act_func = STDPSpike.apply
        self.input_layer = PoissonGenerator()

        self.features, self.classifier = self._make_layers(cfg[self.vgg_name])

    def threshold_init(self, scaling_threshold=1.0, reset_threshold=0.0, thresholds=[], default_threshold=1.0):

        # Initialize thresholds
        self.scaling_threshold = scaling_threshold
        self.reset_threshold = reset_threshold

        self.threshold = {}
        # print('\nThresholds:')

        for pos in range(len(self.features)):
            if isinstance(self.features[pos], nn.Conv2d):
                self.threshold[pos] = thresholds.pop(
                    0) * self.scaling_threshold + self.reset_threshold * default_threshold
            # print('\t Layer{} : {:.2f}'.format(pos, self.threshold[pos]))

        prev = len(self.features)

        for pos in range(len(self.classifier) - 1):
            if isinstance(self.classifier[pos], nn.Linear):
                self.threshold[prev + pos] = thresholds.pop(
                    0) * self.scaling_threshold + self.reset_threshold * default_threshold
            # print('\t Layer{} : {:.2f}'.format(prev+pos, self.threshold[prev+pos]))

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in (cfg):
            stride = 1

            if x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers += [nn.Dropout(0.2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=stride, bias=False),
                           nn.ReLU(inplace=True)
                           ]
                in_channels = x

        features = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(512 * 7 * 7, 4096, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0.2)]
        layers += [nn.Linear(4096, 4096, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0.2)]
        layers += [nn.Linear(4096, self.labels, bias=False)]

        classifer = nn.Sequential(*layers)
        return (features, classifer)

    def network_init(self, update_interval):
        self.update_interval = update_interval

    def neuron_init(self, x):

        self.batch_size = x.size(0)
        self.width = x.size(2)
        self.height = x.size(3)

        self.mem = {}
        self.spike = {}
        self.mask = {}

        for l in range(len(self.features)):

            if isinstance(self.features[l], nn.Conv2d):
                self.mem[l] = torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)

            elif isinstance(self.features[l], nn.Dropout):
                self.mask[l] = self.features[l](torch.ones(self.mem[l - 2].shape))

            elif isinstance(self.features[l], nn.AvgPool2d):
                self.width = self.width // 2
                self.height = self.height // 2

        prev = len(self.features)

        for l in range(len(self.classifier)):

            if isinstance(self.classifier[l], nn.Linear):
                self.mem[prev + l] = torch.zeros(self.batch_size, self.classifier[l].out_features)

            elif isinstance(self.classifier[l], nn.Dropout):
                self.mask[prev + l] = self.classifier[l](torch.ones(self.mem[prev + l - 2].shape))

        self.spike = copy.deepcopy(self.mem)
        for key, values in self.spike.items():
            for value in values:
                value.fill_(-1000)

    def forward(self, x):

        self.neuron_init(x)

        for t in range(self.update_interval):

            out_prev = self.input_layer(x)

            for l in range(len(self.features)):

                if isinstance(self.features[l], (nn.Conv2d)):
                    mem_thr = (self.mem[l] / self.threshold[l]) - 1.0
                    out = self.act_func(mem_thr, (t - 1 - self.spike[l]))
                    rst = self.threshold[l] * (mem_thr > 0).float()
                    self.spike[l] = self.spike[l].masked_fill(out.byte(), t - 1)

                    self.mem[l] = self.leak_mem * self.mem[l] + self.features[l](out_prev) - rst
                    out_prev = out.clone()

                elif isinstance(self.features[l], nn.AvgPool2d):
                    out_prev = self.features[l](out_prev)

                elif isinstance(self.features[l], nn.Dropout):
                    out_prev = out_prev * self.mask[l]

            out_prev = out_prev.reshape(self.batch_size, -1)
            prev = len(self.features)

            for l in range(len(self.classifier) - 1):

                if isinstance(self.classifier[l], (nn.Linear)):
                    mem_thr = (self.mem[prev + l] / self.threshold[prev + l]) - 1.0
                    out = self.act_func(mem_thr, (t - 1 - self.spike[prev + l]))
                    rst = self.threshold[prev + l] * (mem_thr > 0).float()
                    self.spike[prev + l] = self.spike[prev + l].masked_fill(out.byte(), t - 1)

                    self.mem[prev + l] = self.leak_mem * self.mem[prev + l] + self.classifier[l](out_prev) - rst
                    out_prev = out.clone()

                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev = out_prev * self.mask[prev + l]

            # Compute the classification layer outputs
            self.mem[prev + l + 1] = self.mem[prev + l + 1] + self.classifier[l + 1](out_prev)

        return self.mem[prev + l + 1]