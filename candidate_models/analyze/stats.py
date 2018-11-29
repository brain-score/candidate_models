import numpy as np
import scipy.stats

from candidate_models.analyze import DataCollector, align


def compute_correlations(data=None):
    data = data or DataCollector()()
    imagenet = data[data['benchmark'] == 'ImageNet']
    brainscore = data[data['benchmark'] == 'Brain-Score']
    brainscore = brainscore.dropna(axis='rows', subset=['score'])
    imagenet = align(imagenet, brainscore, on='model')

    below70 = imagenet[imagenet['score'] < 70]
    above70 = imagenet[imagenet['score'] >= 70]
    above725 = imagenet[imagenet['score'] >= 72.5]

    def corr(imagenet_selection):
        brainscore_selection = align(brainscore, imagenet_selection, on='model')
        c, p = scipy.stats.pearsonr(imagenet_selection['score'], brainscore_selection['score'])
        return c, p

    below70_corr, below70_p = corr(below70)
    above70_corr, above70_p = corr(above70)
    above725_corr, above725_p = corr(above725)
    print(f"<70% top-1: {below70_corr if below70_p < .05 else 'n.s.'} (p={below70_p})")
    print(f">=70% top-1: {above70_corr if above70_p < .05 else 'n.s.'} (p={above70_p})")
    print(f">=72.5% top-1: {above725_corr if above725_p < .05 else 'n.s.'} (p={above725_p})")


def compute_benchmark_correlations(data=None):
    data = data or DataCollector()()

    # not_mobilenets = [not row['model'].startswith('mobilenet') for _, row in data.iterrows()]
    # data = data[not_mobilenets]

    def corr(a, b):
        c, p = scipy.stats.pearsonr(a, b)
        return c, p

    for benchmark_a, benchmark_b in [
        ('ImageNet', 'dicarlo.Majaj2015.V4'), ('ImageNet', 'dicarlo.Majaj2015.IT'),
        ('ImageNet', 'dicarlo.Rajalingham2018'),
        ('dicarlo.Majaj2015.V4', 'dicarlo.Majaj2015.IT'), ('dicarlo.Majaj2015.V4', 'dicarlo.Rajalingham2018'),
        ('dicarlo.Majaj2015.IT', 'dicarlo.Rajalingham2018')]:
        a, b = data[data['benchmark'] == benchmark_a], data[data['benchmark'] == benchmark_b]
        # clear NaNs and align
        b = b.dropna(axis='rows', subset=['score'])
        a = align(a, b, on='model')
        a = a.dropna(axis='rows', subset=['score'])
        b = align(b, a, on='model')
        # compute
        r, p = corr(a['score'], b['score'])
        print(f"{benchmark_a}, {benchmark_b}: {r if p < .05 else 'n.s.'} (p={p})")
