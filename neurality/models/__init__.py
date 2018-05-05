import argparse
import logging
import sys

import numpy as np

from mkgu.assemblies import NeuroidAssembly
from neurality.assemblies import load_stimulus_set
from neurality.models.graph import get_model_graph, cut_graph
from neurality.models.implementations import model_mappings, prepare_images, create_model, find_images, load_images
from neurality.models.outputs import get_model_outputs
from neurality.models.type import ModelType, get_model_type
from neurality.models.type import ModelType, get_model_type, PYTORCH_SUBMODULE_SEPARATOR
from neurality.storage import store

_logger = logging.getLogger(__name__)


class Defaults(object):
    model_weights = 'imagenet'
    pca_components = 1000
    image_size = 224
    batch_size = 64
    stimulus_set = 'dicarlo.Majaj2015'


@store()
def model_activations(model, layers, stimulus_set, model_weights=Defaults.model_weights,
                      image_size=Defaults.image_size, pca_components=Defaults.pca_components,
                      batch_size=Defaults.batch_size):
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
