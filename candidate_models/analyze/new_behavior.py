import pandas as pd
import scipy.stats
import seaborn
from matplotlib import pyplot

from candidate_models.analyze import align


def plot(save=False):
    data = pd.read_pickle('/braintree/data2/active/users/qbilius//memo//20181126_121838/retest_all.pkl')
    data = data.rename(columns={'r': 'score'})
    data = data[~data['model'].isin(['cornet_z', 'cornet_r', 'cornet_r2'])]

    brainscore_behavior = data[data['kind'] == 'i2n']
    brainscore_behavior['benchmark'] = 'dicarlo.Rajalingham2018'
    brainscore_behavior.to_csv('candidate_models/models/implementations/behavior.csv')
    
    new_behavior = data[data['kind'] == 'i1n']
    new_behavior = align(new_behavior, brainscore_behavior, on='model')

    fig, ax = pyplot.subplots()

    for with_cornet in [False, True]:
        current_brainscore_behavior = brainscore_behavior[[model.startswith('cornet') == with_cornet
                                                           for model in brainscore_behavior['model']]]
        current_new_behavior = align(new_behavior, current_brainscore_behavior, on='model')
        x = current_brainscore_behavior['score']
        y = current_new_behavior['score']
        color = '#808080' if not with_cornet else '#D4145A'
        ax.errorbar(x=x, y=y, linestyle=' ', marker='.', markersize=20, color=color, ecolor=color)

    r, p = scipy.stats.pearsonr(brainscore_behavior['score'], new_behavior['score'])
    assert p <= .05
    ax.text(ax.get_xlim()[1] - .04, ax.get_ylim()[0] + .01, f"r={r:.2f}")
    ax.set_xlabel('Behavioral score, original', fontsize=20)
    ax.set_ylabel(f"Behavioral score, new", fontsize=20)
    seaborn.despine(ax=ax, right=True, top=True)

    if save:
        pyplot.savefig('results/behavior.png')
    return fig


if __name__ == '__main__':
    plot(save=True)
