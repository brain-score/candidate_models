import functools
import os
import seaborn

seaborn.set()
seaborn.set_context('paper', font_scale=2)
seaborn.set_style('whitegrid', {'axes.grid': False})

from matplotlib import pyplot

from candidate_models.analyze.coco import main as neural_generalization
from candidate_models.analyze.cifar_transfer import plot as cifar
from candidate_models.analyze.earlylate import plot_same_layers as earlylate
from candidate_models.analyze.new_behavior import plot as behavior


def main():
    save_formats = ['svg', 'pdf', 'png']
    savedir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'supplement')
    os.makedirs(savedir, exist_ok=True)

    fig_makers = {
        'hvm': (r"$\bf{(a)}$" + ' new neural recordings, \nsame images',
                functools.partial(neural_generalization, benchmark_name='hvm')),
        'coco': (r"$\bf{(b)}$" + ' new neural recordings, \nnew images',
                 functools.partial(neural_generalization, benchmark_name='coco', threshold=.9)),
        'behavior': (r"$\bf{(c)}$" + ' new behavioral recordings, \nnew images',
                     behavior),
        'cifar-100': (r"$\bf{(d)}$" + ' CIFAR-100 transfer\n',
                      cifar),
        'earlylate': ('', earlylate),
    }
    for name, (title, fig_maker) in fig_makers.items():
        fig = fig_maker()
        fig.suptitle(title, fontsize=30, y=1.)
        pyplot.tight_layout()
        for extension in save_formats:
            savepath = os.path.join(savedir, f"{name}.{extension}")
            fig.savefig(savepath, format=extension)
            print("Saved to", savepath)


if __name__ == '__main__':
    main()
