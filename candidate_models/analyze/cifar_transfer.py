import argparse
import logging
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas
import sklearn.linear_model
import sklearn.preprocessing
from tqdm import tqdm

from candidate_models import model_layers
from candidate_models.models import model_activations
from result_caching import store


@store()
def train(model, cifar=10, n_train=40000, n_val=10000):
    df = []

    features = model_activations(model, stimulus_set=f'cifar-{cifar}', layers=model_layers[model][-1:], pca_components=None)
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
    for c in tqdm(np.logspace(-6, 5, 45)):
        reg.set_params(C=c)
        reg.fit(train_feats, train_labels)
        acc = reg.score(val_feats, val_labels)
        df.append({'c': c, 'acc': acc})

    df = pandas.DataFrame(df)
    return df


def test(model, best_c, cifar=10):
    features = model_activations(model, stimulus_set=f'cifar-{cifar}', layers=model_layers[model][-1:], pca_components=None)
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


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet-101_v2')
    parser.add_argument('--cifar', type=int, default=10)
    args = parser.parse_args()
    logging.getLogger(__name__).info(f"Running with args: {args}")

    # c_values = train(args.model, cifar=args.cifar)
    # best_c = c_values.loc[c_values['acc'].idxmax()]['c']
    best_c = .005
    acc = test(args.model, cifar=args.cifar, best_c=best_c)
    print(f"{args.model} on {args.cifar} -> {acc}")
