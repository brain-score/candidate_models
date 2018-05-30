import copy
import functools
import logging
import os

import skimage.transform
from collections import OrderedDict

import h5py
from pathos.pools import ThreadPool as Pool

import numpy as np
from sklearn.decomposition import PCA

from neurality.models.type import get_model_type, ModelType, PYTORCH_SUBMODULE_SEPARATOR

_logger = logging.getLogger(__name__)


def get_model_outputs(model, x, layer_names, preprocess_hack, batch_size=None, pca_components=None):
    if pca_components is not None:
        _logger.info('Pre-computing principal components')
        imagenet_data = _get_imagenet_val(pca_components, image_size=x.shape[-2])
        imagenet_data = _pad(imagenet_data, batch_size)
        imagenet_data = preprocess_hack(imagenet_data)  # wrap imagenet_data in preprocessing
        imagenet_outputs = compute_layer_outputs(model, imagenet_data, layer_names, batch_size=batch_size)
        imagenet_outputs = OrderedDict((layer, outputs[:pca_components]) for layer, outputs in imagenet_outputs.items())
        pca = arrange_layer_outputs(imagenet_outputs, pca_components=pca_components, fit=True)
        arrange_outputs = functools.partial(arrange_layer_outputs, pca_components=pca, fit=False)
    else:
        arrange_outputs = None

    _logger.info('Computing layer outputs')
    outputs = compute_layer_outputs(model, x, layer_names, batch_size, arrange_outputs=arrange_outputs)
    return outputs


def _pad(x, batch_size):
    if x.shape[0] % batch_size == 0:
        return x
    padding = batch_size - (x.shape[0] % batch_size)
    padding = np.zeros([padding, *x.shape[1:]])
    return np.concatenate((x, padding))


def _get_imagenet_val(nimg, image_size):
    n_img_per_class = (nimg - 1) // 1000
    base_idx = np.arange(n_img_per_class).astype(int)
    idx = []
    for i in range(1000):
        idx.extend(50 * i + base_idx)

    for i in range((nimg - 1) % 1000 + 1):
        idx.extend(50 * i + np.array([n_img_per_class]).astype(int))

    imagenet_file = '/braintree/data2/active/users/qbilius/datasets/imagenet2012.hdf5'
    if not os.path.isfile(imagenet_file):
        _logger.error("Imagenet file not found - mocking data")
        return np.random.rand(nimg, image_size, image_size, 3)
    with h5py.File(imagenet_file, 'r') as f:
        ims = np.array([skimage.transform.resize(f['val/images'][i], (image_size, image_size)) for i in idx])
    return ims


def compute_layer_outputs(model, x, layer_names, batch_size, arrange_outputs=None):
    arrange_outputs = arrange_outputs or (lambda x: x)
    model_type = get_model_type(model)
    _compute_layer_outputs = {ModelType.KERAS: compute_layer_outputs_keras,
                              ModelType.PYTORCH: compute_layer_outputs_pytorch,
                              ModelType.SLIM: compute_layer_outputs_slim}[model_type]
    if batch_size is None or not (0 < batch_size < len(x)):
        _logger.debug("Computing all outputs at once")
        outputs = _compute_layer_outputs(layer_names, model, x)
        outputs = arrange_outputs(outputs)
    else:
        outputs = None
        batch_start = 0
        while batch_start < len(x):
            batch_end = min(batch_start + batch_size, len(x))
            _logger.debug('Batch: %d->%d/%d', batch_start, batch_end, len(x))
            batch = x[batch_start:batch_end]
            batch_output = _compute_layer_outputs(layer_names, model, batch)
            batch_output = arrange_outputs(batch_output)
            if outputs is None:
                outputs = copy.copy(batch_output)
            else:
                for layer_name, layer_output in batch_output.items():
                    outputs[layer_name] = np.concatenate((outputs[layer_name], layer_output))
            batch_start = batch_end
    return outputs


def compute_layer_outputs_slim(layer_names, model, images):
    layer_outputs = model.run(images, layer_names)
    return layer_outputs


def compute_layer_outputs_keras(layer_names, model, x):
    from keras import backend as K
    input_tensor = model.input
    layers = [layer for layer in model.layers if layer.name in layer_names]
    layers = sorted(layers, key=lambda layer: layer_names.index(layer.name))
    layer_out_tensors = [layer.output for layer in layers]
    functor = K.function([input_tensor] + [K.learning_phase()], layer_out_tensors)  # evaluate all tensors at once
    layer_outputs = functor([x, 0.])  # 1.: training, 0.: test
    return OrderedDict([(layer_name, layer_output) for layer_name, layer_output
                        in zip([layer.name for layer in layers], layer_outputs)])


def compute_layer_outputs_pytorch(layer_names, model, x):
    layer_results = OrderedDict()

    def walk_pytorch_module(module, layer_name):
        for part in layer_name.split(PYTORCH_SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
        return module

    def store_layer_output(layer_name, output):
        layer_results[layer_name] = output.data.numpy()

    for layer_name in layer_names:
        layer = walk_pytorch_module(model, layer_name)
        layer.register_forward_hook(lambda _layer, _input, output, name=layer_name: store_layer_output(name, output))
    model(x)
    return layer_results


class PCAMock(object):
    def transform(self, x):
        return x


def _arrange(layer_name_output, pca_components, fit=False):
    layer_name, layer_output = layer_name_output
    _logger.debug('Arranging layer output %s (shape %s)', layer_name, str(layer_output.shape))
    layer_output = layer_output.reshape(layer_output.shape[0], -1)

    if fit:
        pca_necessary = pca_components is not None and 0 < pca_components < np.prod(layer_output.shape[1:])
        if pca_necessary:
            assert layer_output.shape[0] >= pca_components, \
                "output has %d components but must have more than %d PCA components" % (
                    layer_output.shape[0], pca_components)
            pca = PCA(n_components=pca_components).fit(layer_output)
            return layer_name, pca
        else:
            return layer_name, PCAMock()
    else:
        pca = pca_components[layer_name]
        return layer_name, pca.transform(layer_output)


def arrange_layer_outputs(outputs, pca_components, fit=False, multiprocessing=False):
    arrange = functools.partial(_arrange, pca_components=pca_components, fit=fit)
    items = outputs.items()
    if multiprocessing:
        with Pool() as pool:
            arranged_outputs = pool.map(arrange, items)
    else:
        arranged_outputs = map(arrange, items)
    return OrderedDict(arranged_outputs)
