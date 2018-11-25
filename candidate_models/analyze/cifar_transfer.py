import pandas as pd
import argparse
import logging
import sys

import numpy as np
import pandas
import scipy.stats
import sklearn.linear_model
import sklearn.preprocessing
from matplotlib import pyplot
from tqdm import tqdm

from candidate_models import model_layers
from candidate_models.analyze import DataCollector, align
from candidate_models.models import model_activations
from result_caching import store


@store()
def train(model, cifar=10, n_train=40000, n_val=10000):
    df = []

    features = model_activations(model, stimulus_set=f'cifar-{cifar}', layers=model_layers[model][-1:],
                                 pca_components=None)
    features = features.sel(type='train').transpose('presentation', 'neuroid')
    label_key = 'fine_label' if cifar == 100 else 'label'
    trainval_feats = features.values
    trainval_labels = features[label_key].values

    # normalize features
    norm = sklearn.preprocessing.StandardScaler()
    # and divide into train / val
    train_feats = norm.fit_transform(trainval_feats[:n_train])
    val_feats = norm.fit_transform(trainval_feats[n_train:n_train + n_val])

    train_labels = trainval_labels[:n_train]
    val_labels = trainval_labels[n_train:n_train + n_val]

    reg = sklearn.linear_model.LogisticRegression(solver='lbfgs',
                                                  multi_class='multinomial',
                                                  warm_start=True,
                                                  n_jobs=-1)
    # find the best C in validation set
    for c in tqdm(5 * np.logspace(-4, 2, 10)):
        reg.set_params(C=c)
        reg.fit(train_feats, train_labels)
        acc = reg.score(val_feats, val_labels)
        df.append({'c': c, 'acc': acc})

    df = pandas.DataFrame(df)
    return df


@store()
def test(model, best_c, cifar=10):
    features = model_activations(model, stimulus_set=f'cifar-{cifar}', layers=model_layers[model][-1:],
                                 pca_components=None)
    features = features.transpose('presentation', 'neuroid')
    label_key = 'fine_label' if cifar == 100 else 'label'
    trainval_feats = features.sel(type='train').values
    trainval_labels = features.sel(type='train')[label_key].values
    test_feats = features.sel(type='test').values
    test_labels = features.sel(type='test')[label_key].values

    # normalize features
    norm = sklearn.preprocessing.StandardScaler()
    trainval_feats = norm.fit_transform(trainval_feats)
    test_feats = norm.fit_transform(test_feats)

    # get accuracy on the test set using the best C we found
    reg = sklearn.linear_model.LogisticRegression(solver='lbfgs',
                                                  multi_class='multinomial',
                                                  C=best_c,
                                                  n_jobs=-1)
    reg.fit(trainval_feats, trainval_labels)
    acc = reg.score(test_feats, test_labels)
    return acc


def get_cifar_scores(cifar=100):
    result = []
    for model in ["cornet_z", "cornet_r", "cornet_r2", "cornet_s",
                  "alexnet",
                  # "squeezenet1_0", "squeezenet1_1",
                  "xception",
                  "densenet-121", "densenet-169", "densenet-201",
                  "inception_v1", "inception_v2", #"inception_v3", "inception_v4",
                  # "inception_resnet_v2",
                  "resnet-18", "resnet-34", "resnet-50_v2", "resnet-101_v2", "resnet-152_v2",
                  "vgg-16", "vgg-19",
                  "nasnet_mobile", "nasnet_large", "pnasnet_large",
                  "mobilenet_v1_1.0_224", "mobilenet_v1_1.0_192", "mobilenet_v1_1.0_160", "mobilenet_v1_1.0_128",
                  "mobilenet_v1_0.75_224", "mobilenet_v1_0.75_192", "mobilenet_v1_0.75_160", "mobilenet_v1_0.75_128",
                  "mobilenet_v1_0.5_224", "mobilenet_v1_0.5_192", "mobilenet_v1_0.5_160", "mobilenet_v1_0.5_128",
                  "mobilenet_v1_0.25_224", "mobilenet_v1_0.25_192", "mobilenet_v1_0.25_160", "mobilenet_v1_0.25_128",
                  # "mobilenet_v2_1.4_224", "mobilenet_v2_1.3_224", "mobilenet_v2_1.0_224", "mobilenet_v2_1.0_192",
                  # "mobilenet_v2_1.0_160", "mobilenet_v2_1.0_128", "mobilenet_v2_1.0_96", "mobilenet_v2_0.75_224",
                  # "mobilenet_v2_0.75_192", "mobilenet_v2_0.75_160", "mobilenet_v2_0.75_128", "mobilenet_v2_0.75_96",
                  # "mobilenet_v2_0.5_224", "mobilenet_v2_0.5_192", "mobilenet_v2_0.5_160", "mobilenet_v2_0.5_128",
                  # "mobilenet_v2_0.5_96", "mobilenet_v2_0.35_224", "mobilenet_v2_0.35_192", "mobilenet_v2_0.35_160",
                  # "mobilenet_v2_0.35_128", "mobilenet_v2_0.35_96"
                  ]:
        c_values = train(model, cifar=cifar)
        best_c = c_values.loc[c_values['acc'].idxmax()]['c']
        acc = test(model, cifar=cifar, best_c=best_c)
        result.append({'model': model, 'score': acc, 'benchmark': f"cifar-{cifar}"})
    result = pd.DataFrame(result)
    return result


def plot():
    cifar_scores = get_cifar_scores()
    brain_scores = DataCollector()()
    brain_scores = brain_scores[brain_scores['benchmark'] == 'Brain-Score']
    brain_scores = align(brain_scores, cifar_scores, on='model')
    x, y = brain_scores['score'], cifar_scores['score']
    colors = ['r' if model.startswith('cornet') else 'b' for model in brain_scores['model'].values]
    x, y, colors = zip(*[(_x, _y, color) for _x, _y, color in zip(x, y, colors)
                         if not np.isnan(_x) and not np.isnan(_y)])
    pyplot.scatter(x, y, color=colors)

    r, p = scipy.stats.pearsonr(x, y)
    assert p <= .05
    pyplot.text(pyplot.xlim()[1] - .01, pyplot.ylim()[0], f"r={r:.2f}")

    pyplot.xlabel("Brain-Score")
    pyplot.ylabel("CIFAR-100 Accuracy")
    pyplot.savefig('results/cifar-100.png')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet-101_v2')
    parser.add_argument('--cifar', type=int, default=10)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    logging.getLogger(__name__).info(f"Running with args: {args}")

    if args.plot:
        plot()
        sys.exit(0)

    c_values = train(args.model, cifar=args.cifar)
    best_c = c_values.loc[c_values['acc'].idxmax()]['c']
    # best_c = .005
    acc = test(args.model, cifar=args.cifar, best_c=best_c)
    print(f"{args.model} on {args.cifar} -> {acc} (c={best_c}")
