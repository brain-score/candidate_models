
import numpy as np


def gabor_kernel(frequency,  sigma_x, sigma_y, theta=0, offset=0, ks=61):

    w = np.floor(ks/2)
    y, x = np.mgrid[-w:w + 1, -w:w + 1]

    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    g = np.zeros(y.shape)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.cos(2 * np.pi * frequency * rotx + offset)

    return g


def gauss_kernel(sigma, ks=61):

    w = np.floor(ks/2)
    y, x = np.mgrid[-w:w + 1, -w:w + 1]

    g = np.zeros(y.shape)
    g[:] = np.exp(-0.5 * ((x ** 2 + y ** 2) / sigma ** 2))
    g /= 2 * np.pi * sigma ** 2

    return g


def sample_dist(hist, bins, ns, scale='linear'):
    rand_sample = np.random.rand(ns)
    if scale == 'linear':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), bins)
    elif scale == 'log2':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log2(bins))
        rand_sample = 2**rand_sample
    elif scale == 'log10':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log10(bins))
        rand_sample = 10**rand_sample
    return rand_sample

