import os

import numpy as np


def compute_behavioral_differences():
    diff('pnasnet_large')


def diff(model):
    data = np.load(os.path.join(os.path.dirname(__file__), '{}.npy'.format(model)))
    model_responses, human_responses = data[:, 0], data[:, 1]
    pairs = zip(human_responses, model_responses)
    diffs = np.array([human - model for human, model in pairs])
    num_images = len(model_responses)
    equal = sum((diffs <= 1) & (diffs >= -1))
    human_better = sum(diffs > 1)
    model_better = sum(diffs < -1)
    print(f"Total number of images: {len(model_responses)} {model} responses and {len(human_responses)} human responses")
    print(f"Roughly equal: {equal} ({equal / num_images * 100:.2f}%)")
    print(f"Human better: {human_better} ({human_better / num_images * 100:.2f}%)")
    print(f"Model better: {model_better} ({model_better / num_images * 100:.2f}%)")
    print(f"Sanity check sum: {equal + human_better + model_better}")


if __name__ == '__main__':
    compute_behavioral_differences()
