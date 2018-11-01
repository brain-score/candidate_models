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
    layers = {}

    def best_layer(score):
        argmax = score.sel(aggregation='center', _apply_raw=False).argmax('layer')  # choose best layer
        best_layer = score['layer'][argmax.values]
        score = score.sel(layer=best_layer)
        layers[score['image_size'].values.tolist()] = best_layer.values.tolist()
        return score

    scores = scores.groupby('image_size').apply(best_layer)
    x = scores['image_size'].values
    y = scores.sel(aggregation='center').values
    error = scores.sel(aggregation='error').values

    fig, ax = pyplot.subplots()
    shaded_errorbar(x, y, error, ax=ax)
    for _x, _y in zip(x, y):
        pyplot.text(x=_x, y=_y, s=layers[_x])
    pyplot.xlabel('image size')
    pyplot.ylabel('best r')
    pyplot.savefig(f'results/image_size-{benchmark}-{model}.png')
    return fig


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main('vgg-19', benchmark='tolias.Cadena2017')
    main('alexnet', benchmark='tolias.Cadena2017')
