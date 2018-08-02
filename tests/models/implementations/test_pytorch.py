import os

from candidate_models.models.implementations.pytorch import PytorchModel


class TestLoadImage:
    def test_rgb(self):
        img = PytorchModel._load_image(None, os.path.join(os.path.dirname(__file__), 'rgb.jpg'))
        assert img.mode == 'RGB'

    def test_grayscale(self):
        img = PytorchModel._load_image(None, os.path.join(os.path.dirname(__file__), 'grayscale.png'))
        assert img.mode == 'RGB'


class TestGraph:
    def test_alexnet(self):
        model = PytorchModel('alexnet', weights=None, batch_size=64, image_size=224)
        graph = model.graph()
        assert 20 == len(graph.nodes)
        assert 8 == len([node_name for node_name, node in graph.nodes.items()
                         if 'conv' in node['type'].__name__.lower() or 'linear' in node['type'].__name__.lower()])
