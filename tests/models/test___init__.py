import numpy as np
from tests.models import patch_imagenet

from candidate_models.models import model_activations
from candidate_models.models import model_multi_activations
from candidate_models.models.implementations import model_layers


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


class TestNewModel:
    def test_pytorch(self):
        import torch
        from torch import nn
        from candidate_models.models.implementations.pytorch import PytorchModel

        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
                self.relu1 = torch.nn.ReLU()
                linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
                self.linear = torch.nn.Linear(int(linear_input_size), 1000)
                self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                x = self.relu2(x)
                return x

        class MyModelWrapper(PytorchModel):
            def __init__(self, *args, **kwargs):
                super().__init__('mymodel', *args, **kwargs)

            def _create_model(self, weights):
                my_model = MyModel()
                assert weights is None  # weight loading would go here
                return my_model

        activations = model_activations(model=MyModelWrapper, model_identifier='test_pytorch',
                                        layers=['linear', 'relu2'], weights=None, pca_components=None)
        assert activations is not None
        assert len(activations['neuroid']) == 1000 + 1000

    def test_tensorflow_slim(self):
        import tensorflow as tf
        from preprocessing import vgg_preprocessing
        slim = tf.contrib.slim
        from candidate_models.models.implementations.tensorflow_slim import TensorflowSlimModel

        class MyModelWrapper(TensorflowSlimModel):
            def _create_inputs(self, batch_size, image_size):
                inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
                preprocess_image = vgg_preprocessing.preprocess_image
                return tf.map_fn(lambda image: preprocess_image(tf.image.convert_image_dtype(image, dtype=tf.uint8),
                                                                image_size, image_size), inputs)

            def _create_model(self, inputs):
                with tf.variable_scope('my_model', values=[inputs]) as sc:
                    end_points_collection = sc.original_name_scope + '_end_points'
                    # Collect outputs for conv2d, fully_connected and max_pool2d.
                    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                        outputs_collections=[end_points_collection]):
                        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
                        net = slim.max_pool2d(net, [5, 5], 5, scope='pool1')
                        net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                        return net, end_points

            def _restore(self, weights):
                assert weights is None
                init = tf.initialize_all_variables()
                self._sess.run(init)

        activations = model_activations(model=MyModelWrapper, model_identifier='test_tensorflow_slim',
                                        layers=['my_model/pool2'],
                                        weights=None, pca_components=None)
        assert activations is not None
        assert len(activations['neuroid']) == 4 * 4 * 64


class TestModelActivations:
    def test_single_layer(self, mocker):
        patch_imagenet(mocker)
        activations = model_activations(model='alexnet', layers=['features.12'])
        assert unique_preserved_order(activations['layer']) == 'features.12'

    def test_two_layers(self, mocker):
        patch_imagenet(mocker)
        activations = model_activations(model='alexnet', layers=['features.12', 'classifier.2'])
        np.testing.assert_array_equal(unique_preserved_order(activations['layer']), ['features.12', 'classifier.2'])

    def test_all_layers(self, mocker):
        patch_imagenet(mocker)
        layers = model_layers['alexnet']
        activations = model_activations(model='alexnet', layers=layers)
        np.testing.assert_array_equal(unique_preserved_order(activations['layer']), layers)


class TestModelMultiActivations:
    def test_single_layer(self, mocker):
        patch_imagenet(mocker)
        activations = model_multi_activations(model='alexnet', multi_layers=[['features.12']])
        assert unique_preserved_order(activations['layer']) == 'features.12'

    def test_combine_two(self, mocker):
        patch_imagenet(mocker)
        activations = model_multi_activations(model='alexnet', multi_layers=[['features.12', 'classifier.2']])
        assert len(np.unique(activations['layer'])) == 1
        assert unique_preserved_order(activations['layer']) == 'features.12,classifier.2'

    def test_combine_two_two(self, mocker):
        patch_imagenet(mocker)
        activations = model_multi_activations(model='alexnet',
                                              multi_layers=[['features.2', 'features.5'],
                                                            ['features.12', 'classifier.2']])
        assert len(np.unique(activations['layer'])) == 2
        np.testing.assert_array_equal(unique_preserved_order(activations['layer']),
                                      ['features.2,features.5', 'features.12,classifier.2'])
