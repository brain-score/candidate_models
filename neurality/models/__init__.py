import argparse
import logging
import sys

import numpy as np
import xarray as xr

from mkgu.assemblies import NeuroidAssembly, merge_data_arrays
from mkgu.metrics import subset, get_modified_coords
from neurality.assemblies import load_stimulus_set
from neurality.models.graph import get_model_graph, cut_graph
from neurality.models.implementations import model_mappings, prepare_images, create_model, find_images, load_images
from neurality.models.outputs import get_model_outputs
from neurality.models.type import ModelType, get_model_type
from neurality.models.type import ModelType, get_model_type, PYTORCH_SUBMODULE_SEPARATOR
from neurality.storage import store_xarray

_logger = logging.getLogger(__name__)


class Defaults(object):
    model_weights = 'imagenet'
    pca_components = 1000
    image_size = 224
    batch_size = 64
    stimulus_set = 'dicarlo.Majaj2015'


def model_multi_activations(model, multi_layers, stimulus_set=Defaults.stimulus_set,
                            model_weights=Defaults.model_weights,
                            image_size=Defaults.image_size, pca_components=Defaults.pca_components,
                            batch_size=Defaults.batch_size):
    single_layers = []
    for layers in multi_layers:
        if isinstance(layers, str):
            single_layers.append(layers)
        else:
            for layer in layers:
                single_layers.append(layer)
    single_layers = list(set(single_layers))
    single_layer_activations = model_activations(model, single_layers, stimulus_set, model_weights=model_weights,
                                                 image_size=image_size, pca_components=pca_components,
                                                 batch_size=batch_size)

    multi_layer_activations = []
    for layers in multi_layers:
        layers_target = xr.DataArray(np.full(len(layers), np.nan), coords={'layer': layers}, dims=['layer'])
        layers_target = layers_target.stack(neuroid=['layer'])
        layers_activations = subset(single_layer_activations, layers_target, dims_must_match=False)

        # at this point, layers_activations are concatenated across layers
        # BUT they will be disconnected again later due to layer being an adjacent coordinate.
        # we set `layer` to the concatenated coords and keep the original `layer` in another coord.
        noncombined_layers = layers_activations['layer'].dims, layers_activations['layer'].values

        def modify_coord(name, dims, values):
            # we can't build a list here because xarray won't allow that later on. instead, combine with string join
            return name, (dims, values if name != 'layer' else np.repeat(combine_layers_xarray(layers), len(values)))

        coords = get_modified_coords(layers_activations, modify_coord)
        coords['noncombined_layer'] = noncombined_layers
        layers_activations = NeuroidAssembly(layers_activations.values, coords=coords, dims=layers_activations.dims)

        multi_layer_activations.append(layers_activations)
    return merge_data_arrays(multi_layer_activations)


def combine_layers_xarray(layers):
    return ",".join(layers)


def split_layers_xarray(layers):
    return layers.split(",")


@store_xarray(identifier_ignore=['batch_size', 'layers'], combine_fields={'layers': 'layer'})
def model_activations(model, layers, stimulus_set=Defaults.stimulus_set, model_weights=Defaults.model_weights,
                      image_size=Defaults.image_size, pca_components=Defaults.pca_components,
                      batch_size=Defaults.batch_size):
    for layer in layers:
        if not isinstance(layer, str):
            raise ValueError("This method does not allow multi-layer activations. "
                             "Use model_multi_activations instead.")
    _logger.info('Loading stimuli')
    stimulus_set = load_stimulus_set(stimulus_set)
    stimuli_paths = list(map(stimulus_set.get_image, stimulus_set['hash_id']))

    _logger.info('Creating model')
    model, preprocess_input = create_model(model, model_weights=model_weights, image_size=image_size)
    model_type = get_model_type(model)
    _verify_model_layers(model, layers)

    _logger.info('Computing activations')
    images = load_images(image_filepaths=stimuli_paths, preprocess_input=preprocess_input,
                         model_type=model_type, image_size=image_size)
    layer_activations = get_model_outputs(model, images, layers,
                                          batch_size=batch_size, pca_components=pca_components)

    _logger.info('Packaging into assembly')
    activations = np.array(list(layer_activations.values())).transpose([1, 0, 2])  # images x layer x activations
    assert activations.shape[0] == len(stimulus_set)
    assert activations.shape[1] == len(layer_activations)
    layers = np.array(list(layer_activations.keys()))
    layers = np.repeat(layers[:, np.newaxis], repeats=activations.shape[-1], axis=1)

    activations = np.reshape(activations, [activations.shape[0], np.prod(activations.shape[1:])])
    layers = np.reshape(layers, [np.prod(activations.shape[1:])])

    model_assembly = NeuroidAssembly(
        activations,
        coords={'image_id': stimulus_set['hash_id'],
                'neuroid_id': list(range(activations.shape[1]))},
        dims=['image_id', 'neuroid_id']
    )
    model_assembly['layer'] = 'neuroid_id', layers
    for column in stimulus_set.columns:
        model_assembly[column] = 'image_id', stimulus_set[column]
    model_assembly = model_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))
    return model_assembly


def model_graph(model, layers):
    graph = get_model_graph(model)
    return cut_graph(graph, keep_nodes=layers, fill_in=True)


def _verify_model_layers(model, layer_names):
    model_type = get_model_type(model)
    _verify_model = {ModelType.KERAS: _verify_model_layers_keras,
                     ModelType.PYTORCH: _verify_model_layers_pytorch}[model_type]
    _verify_model(model, layer_names)


def _verify_model_layers_pytorch(model, layer_names):
    def collect_pytorch_layer_names(module, parent_module_parts):
        result = []
        for submodule_name, submodule in module._modules.items():
            if not hasattr(submodule, '_modules') or len(submodule._modules) == 0:
                result.append(PYTORCH_SUBMODULE_SEPARATOR.join(parent_module_parts + [submodule_name]))
            else:
                result += collect_pytorch_layer_names(submodule, parent_module_parts + [submodule_name])
        return result

    nonexisting_layers = set(layer_names) - set(collect_pytorch_layer_names(model, []))
    assert len(nonexisting_layers) == 0, "Layers not found in PyTorch model: %s" % str(nonexisting_layers)


def _verify_model_layers_keras(model, layer_names):
    model_layers = [layer.name for layer in model.layers]
    nonexisting_layers = set(layer_names) - set(model_layers)
    assert len(nonexisting_layers) == 0, "Layers not found in keras model: {} (model layers: {})".format(
        nonexisting_layers, model_layers)


def main():
    parser = argparse.ArgumentParser('model comparison')
    parser.add_argument('--model', type=str, required=True, choices=list(model_mappings.keys()))
    parser.add_argument('--model_weights', type=str, default=Defaults.model_weights)
    parser.add_argument('--no-model_weights', action='store_const', const=None, dest='model_weights')
    parser.add_argument('--layers', nargs='+', required=True)
    parser.add_argument('--pca', type=int, default=Defaults.pca_components,
                        help='Number of components to reduce the flattened features to')
    parser.add_argument('--image_size', type=int, default=Defaults.image_size)
    parser.add_argument('--stimulus_set', type=str, default=Defaults.stimulus_set)
    parser.add_argument('--batch_size', type=int, default=Defaults.batch_size)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    _logger.info("Running with args %s", vars(args))

    model_activations(model=args.model, layers=args.layers, image_size=args.image_size,
                      stimulus_set=args.stimulus_set, pca_components=args.pca, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
