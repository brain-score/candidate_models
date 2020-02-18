
import numpy as np
from .cornetv1_utils import sample_dist


def generate_filter_param(features, seed=123, rand_flag=False):
    if rand_flag:
        nx_bins = np.array([0.2, 1.0])
        nx_dist = np.array([1])

        n_ratio_bins = np.array([0.5, 4.])
        n_ratio_dist = np.array([1])

        spont_bins = np.array([0, 40])
        spont_dist = np.array([1])

        k_exc_bins = np.array([4.64158883, 464.15888336])
        k_exc_dist = np.array([1])

        ori_bins = np.array([0, 180])
        ori_dist = np.array([1])

        sf_bins = np.array([1.0, 8])
        sf_dist = np.array([1])

        ssi_bins = np.array([0, 1.0])
        ssi_dist = np.array([1])

    else:
        # Ringach 2002b
        nx_bins = np.array([0.2, 0.4, 0.6,  0.8, 1.])
        nx_dist = np.array([47, 22, 4, 1])
        nx_dist = nx_dist / nx_dist.sum()
        n_ratio_bins = np.array([0.5, 0.70710678, 1., 1.41421356, 2.,2.82842712, 4. ])
        n_ratio_dist = np.array([8, 15, 30, 24,  7, 2])
        n_ratio_dist = n_ratio_dist / n_ratio_dist.sum()

        # Ringach 2002a
        spont_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
        spont_dist = np.array([227, 35, 26, 12, 4, 1, 1, 2])
        spont_dist = spont_dist / spont_dist.sum()

        # Ringach 2002a
        k_exc_bins = np.array([4.64158883, 10., 21.5443469, 46.41588834, 100., 215.443469, 464.15888336])
        k_exc_dist = np.array([29, 78, 112, 70, 15, 3, ])
        k_exc_dist = k_exc_dist / k_exc_dist.sum()

        # DeValois 1982a
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        ori_dist = np.array([66, 49, 77, 54])
        ori_dist = ori_dist / ori_dist.sum()

        # DeValois 1982b
        sf_bins = np.array([1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([17, 34, 38, 35, 50, 24])
        sf_dist = sf_dist / sf_dist.sum()

        # Cavanaugh 2002
        ssi_bins = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ssi_dist = np.array([66, 44, 38, 23, 16])
        ssi_dist = ssi_dist / ssi_dist.sum()

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    # Generates random sample
    np.random.seed(seed)
    nx = sample_dist(nx_dist, nx_bins, features)
    n_ratio = sample_dist(n_ratio_dist, n_ratio_bins, features, scale='log2')
    spont = sample_dist(spont_dist, spont_bins, features)
    k_exc = sample_dist(k_exc_dist, k_exc_bins, features, scale='log10')
    # k_inh = sample_dist(k_inh_dist, k_inh_bins, features)
    ori = sample_dist(ori_dist, ori_bins, features)
    ori[ori < 0] = ori[ori < 0] + 180
    sf = sample_dist(sf_dist, sf_bins, features, scale='log2')
    ssi = sample_dist(ssi_dist, ssi_bins, features)
    k_inh = (1 / (1 - ssi) - 1)
    phase = sample_dist(phase_dist, phase_bins, features)

    return nx, n_ratio, spont, k_exc, k_inh, ori, sf, phase
