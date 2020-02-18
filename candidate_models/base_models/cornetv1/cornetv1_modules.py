
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .cornetv1_utils import gabor_kernel, gauss_kernel


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class GFBc(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)

        # Param instatiations
        self.weight = torch.zeros((out_channels, 1, kernel_size, kernel_size))

    def forward(self, x):
        for c in range(self.in_channels):
            if c == 0:
                out = F.conv2d(x[:, c:c+1], self.weight, None, self.stride, self.padding)
                out = out.view((out.shape[0], 1, out.shape[1], out.shape[2], out.shape[3]))
            else:
                out = torch.cat((out, F.conv2d(x[:, c:c+1], self.weight, None, self.stride, self.padding).
                                 view((out.shape[0], 1, out.shape[2], out.shape[3], out.shape[4]))), 1)
        return out

    def initialize(self, sf, theta, sigx, sigy, phase):  # removed k
        for i in range(self.out_channels):
            gk = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i], theta=theta[i], offset=phase[i],
                              ks=self.kernel_size[0]) * 2
            self.weight[i, 0] = torch.Tensor(gk)
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class GaussianAvgPool2d(nn.Module):
    def __init__(self, in_channels, sig, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)
        self.sigma = sig

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding) / self.in_channels

    def initialize(self):
        gaussk = gauss_kernel(sigma=self.sigma, ks=self.kernel_size[0])
        self.weight = torch.Tensor(gaussk).expand(1, self.in_channels, -1, -1)
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class V1_c(nn.Module):
    def __init__(self, in_channels, simple_channels, complex_channels, sf, theta, sigx, sigy, k_inh,
                 sig_div=3.5, phase=None, k_exc=None, spont=None, ksize=59, ksize_div=13, stride=4):
        super().__init__()

        self.in_channels = in_channels
        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        out_channels = (simple_channels + complex_channels)

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.k_inh = nn.Parameter(torch.Tensor(k_inh).view(out_channels, 1, 1), requires_grad=False)
        self.sig_div = sig_div
        if phase is None:
            self.phase = np.zeros(out_channels)
        else:
            self.phase = phase
        if k_exc is None:
            self.k_exc = nn.Parameter(torch.ones(out_channels, 1, 1), requires_grad=False)
        else:
            self.k_exc = nn.Parameter(torch.Tensor(k_exc).view(out_channels, 1, 1), requires_grad=False)
        if spont is None:
            self.spont = nn.Parameter(torch.zeros(out_channels, 1, 1), requires_grad=False)
        else:
            self.spont = nn.Parameter(torch.Tensor(spont).view(out_channels, 1, 1), requires_grad=False)

        # self.input = Identity()
        self.simple_conv_q0 = GFBc(in_channels, simple_channels + complex_channels, ksize, stride)
        self.simple_conv_q1 = GFBc(in_channels, simple_channels + complex_channels, ksize, stride)

        self.simple_nl = nn.ReLU()
        self.complex_nl = Identity()

        self.exc = Identity()
        self.div = GaussianAvgPool2d(1, sig=self.sig_div, kernel_size=ksize_div, stride=1)
        self.inh = Identity()
        self.output = Identity()

        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2)
        self.div.initialize()

    def forward(self, inp):
        # Returns tensor: Batch x 3 x features x H x W
        simple_conv_q0 = self.simple_conv_q0(inp)
        simple_conv_q1 = self.simple_conv_q1(inp)

        simple_nl = self.simple_nl(simple_conv_q0)
        complex_nl = self.complex_nl(torch.sqrt(simple_conv_q0 ** 2 + simple_conv_q1 ** 2) / np.sqrt(2))

        exc_resp = torch.cat((simple_nl[:, :, 0:self.simple_channels, :, :],
                              complex_nl[:, :, self.simple_channels:, :, :]), 2)
        exc_resp = self.exc(self.k_exc * exc_resp)

        # Returns tensor: Batch x features x H x W
        inh_resp = self.inh(self.k_inh * self.div(torch.mean(torch.mean(complex_nl, dim=1, keepdim=True),
                                                             dim=2, keepdim=False)))

        # Returns tensor: Batch x 3 x features x H x W
        out_resp = self.output(exc_resp / (1 + inh_resp.view((inh_resp.shape[0], 1, inh_resp.shape[1],
                                                              inh_resp.shape[2], inh_resp.shape[3]))) + self.spont)

        # Returns tensor: Batch x features x H x W
        out_resp = out_resp.reshape((out_resp.shape[0], out_resp.shape[1] * out_resp.shape[2], out_resp.shape[3],
                                 out_resp.shape[4]))
        return out_resp


