import networkx as nx
from collections import OrderedDict

import numpy as np

from candidate_models import cut_graph, model_graph, combine_graph
from candidate_models.models.implementations import vgg16


class TestGraphComparison:
    def assert_graphs_equal(self, graph1, graph2):
        np.testing.assert_array_equal(list(graph1.nodes), list(graph2.nodes),
                                      "Graph nodes are not equal: {} <> {}".format(graph1.nodes, graph2.nodes))
        assert nx.is_isomorphic(graph1, graph2), \
            "Graphs are not isomorphic: {} <> {}".format(graph1.edges, graph2.edges)


class TestModelGraph(TestGraphComparison):
    def test_vgg16(self):
        model = vgg16(224)[0]
        layers = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool', 'fc1', 'fc2']
        target_graph = nx.DiGraph()
        for layer1, layer2 in zip(layers, layers[1:]):
            target_graph.add_edge(layer1, layer2)

        graph = model_graph(model, layers=layers)
        self.assert_graphs_equal(graph, target_graph)


class TestCombineGraph(TestGraphComparison):
    def test_unary_mapping(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        mapping = OrderedDict([('1', ['A']), ('2', ['B'])])
        combined_graph = combine_graph(graph, mapping)

        target_graph = nx.DiGraph()
        target_graph.add_edge('1', '2')
        self.assert_graphs_equal(combined_graph, target_graph)

    def test_combine_end(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        mapping = OrderedDict([('1', ['A']), ('2', ['B', 'C'])])
        combined_graph = combine_graph(graph, mapping)

        target_graph = nx.DiGraph()
        target_graph.add_edge('1', '2')
        self.assert_graphs_equal(combined_graph, target_graph)

    def test_combine_skip(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        graph.add_edge('A', 'D')
        mapping = OrderedDict([('1', ['A']), ('2', ['B']), ('3', ['C', 'D'])])
        combined_graph = combine_graph(graph, mapping)

        target_graph = nx.DiGraph()
        target_graph.add_edge('1', '2')
        target_graph.add_edge('2', '3')
        target_graph.add_edge('1', '3')
        self.assert_graphs_equal(combined_graph, target_graph)


class TestCutGraph(TestGraphComparison):
    def test_cut_nothing(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        _cut_graph = cut_graph(graph, keep_nodes=['A', 'B'])
        self.assert_graphs_equal(_cut_graph, graph)

    def test_2nodes(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        _cut_graph = cut_graph(graph, keep_nodes=['A'])

        target_graph = nx.DiGraph()
        target_graph.add_node('A')
        self.assert_graphs_equal(_cut_graph, target_graph)

    def test_2nodes_fill_in(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        _cut_graph = cut_graph(graph, keep_nodes=['A'], fill_in=True)

        target_graph = nx.DiGraph()
        target_graph.add_node('A')
        self.assert_graphs_equal(_cut_graph, target_graph)

    def test_4nodes(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        graph.add_edge('A', 'D')
        graph.add_edge('B', 'C')
        _cut_graph = cut_graph(graph, keep_nodes=['B', 'C', 'D'])

        target_graph = nx.DiGraph()
        target_graph.add_edge('B', 'C')
        target_graph.add_edge('C', 'D')
        target_graph.add_edge('B', 'C')
        self.assert_graphs_equal(_cut_graph, target_graph)

    def test_4nodes_fill_in(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        _cut_graph = cut_graph(graph, keep_nodes=['A', 'D'], fill_in=True)

        target_graph = nx.DiGraph()
        target_graph.add_edge('A', 'D')
        self.assert_graphs_equal(_cut_graph, target_graph)
