import numpy as np
import scipy.stats

from candidate_models.analyze import DataCollector


def compute_correlations(data=None):
    data = data or DataCollector()()
    below70 = data[data['performance'] < 70]
    above70 = data[data['performance'] >= 70]
    above725 = data[data['performance'] >= 72.5]

    def corr(d):
        c, p = scipy.stats.pearsonr(d['performance'], d['brain-score'])
        return c, p

    below70_corr, below70_p = corr(below70)
    above70_corr, above70_p = corr(above70)
    above725_corr, above725_p = corr(above725)
    print(f"<70% top-1: {below70_corr} (p={below70_p})")
    print(f">=70% top-1: {above70_corr} (p={above70_p})")
    print(f">=72.5% top-1: {above725_corr} (p={above725_p})")


def compute_benchmark_correlations(data=None):
    data = data or DataCollector()()
    data = data[~np.isnan(data['V4']) & ~np.isnan(data['IT'])
                & ~np.isnan(data['behavior'])]
    not_mobilenets = [not row['model'].startswith('mobilenet') for _, row in data.iterrows()]
    data = data[not_mobilenets]
    v4 = data['V4']
    it = data['IT']
    behavior = data['behavior']

    def corr(a, b):
        c, p = scipy.stats.pearsonr(a, b)
        return c, p

    v4_b_corr, v4_b_p = corr(v4, behavior)
    it_b_corr, it_b_p = corr(it, behavior)
    v4_it_corr, v4_it_p = corr(v4, it)
    print(f"v4, behavior: {v4_b_corr} (p={v4_b_p})")
    print(f"it, behavior: {it_b_corr} (p={it_b_p})")
    print(f"v4, it: {v4_it_corr} (p={v4_it_p})")
