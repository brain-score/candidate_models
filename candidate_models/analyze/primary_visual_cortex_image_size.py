import logging
import sys

from matplotlib import pyplot

from brainscore.metrics import Score
from candidate_models import score_model
from candidate_models.analyze import shaded_errorbar


def main(model, benchmark, image_sizes=(48, 96, 128, 224, 320)):
    scores = []
    for image_size in image_sizes:
        # use untrained models for now, not sure how to load weights for smaller input size.
        # it should work, because filter sizes are probably unchanged
        score = score_model(model=model, weights=None, image_size=image_size, benchmark=benchmark)
        score = score.expand_dims('image_size')
        score['image_size'] = [image_size]
        scores.append(score)
    scores = Score.merge(*scores)

    fig, ax = pyplot.subplots()
    for image_size in scores['image_size'].values:
        size_scores = scores.sel(image_size=image_size)
        x = size_scores['layer'].values
        y = size_scores.sel(aggregation='center').values
        error = size_scores.sel(aggregation='error').values

        shaded_errorbar(x, y, error, ax=ax, label=image_size)
    pyplot.legend(scores['image_size'].values)
    pyplot.xlabel('image size')
    pyplot.ylabel('best r')
    pyplot.xticks(scores['layer'].values, rotation=45)
    pyplot.tight_layout()
    pyplot.savefig(f'results/image_size-{benchmark}-{model}.png')
    return fig


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main('vgg-19', benchmark='tolias.Cadena2017')
    main('alexnet', benchmark='tolias.Cadena2017')
