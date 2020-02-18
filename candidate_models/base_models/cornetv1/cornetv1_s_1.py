
from collections import OrderedDict
from torch import nn
from .cornetv1_modules import Identity, Flatten, CORblock_S, V1_c
from .cornetv1_params import generate_filter_param
import numpy as np


def CORnetV1_S(seed=1):
    image_size = 224
    visual_degrees = 8

    scale = 4
    rand_param = False
    inh_mult = 36
    fs = 0
    fc = scale * 64
    ks = 59
    kpool = 13

    nx, n_ratio, _, _, k_inh, theta, sf, _ = generate_filter_param(fs+fc, seed, rand_param)

    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = sigx * n_ratio
    k_inh = k_inh * inh_mult
    theta = theta/180 * np.pi

    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('gfb', V1_c(3, 0, fc, sf=sf, theta=theta, sigx=sigx, sigy=sigy, k_inh=k_inh, ksize=ks, ksize_div=kpool)),
            ('norm_gfb', nn.BatchNorm2d(fc * 3)),
            ('conv_btnk', nn.Conv2d(fc * 3, 64, kernel_size=1, stride=1, bias=False)),
            ('norm_btnk', nn.BatchNorm2d(64)),
            ('nonlin_btnk', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model



