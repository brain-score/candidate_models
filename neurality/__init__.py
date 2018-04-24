import logging

from neurality import models
from neurality.assemblies import load_neural_benchmark
from neurality.models import model_activations, model_graph
from neurality.models.graph import combine_graph, cut_graph
from neurality.plot import plot_layer_correlations, plot_scores, results_dir
from neurality.storage import get_function_identifier, store

logger = logging.getLogger(__name__)


class Defaults(object):
    neural_data = 'dicarlo.majaj2015'


@store()
def score_model(model, layers, neural_data=Defaults.neural_data, model_weights=models.Defaults.model_weights):
    physiology_score = score_physiology(model=model, layers=layers, neural_data=neural_data,
                                        model_weights=model_weights)
    anatomy_score = score_anatomy(model, physiology_score.mapping)
    return [physiology_score, anatomy_score]


def score_physiology(model, layers, neural_data=Defaults.neural_data, model_weights=models.Defaults.model_weights):
    logger.info('Computing activations')
    activations = model_activations(model=model, model_weights=model_weights, layers=layers, stimulus_set=neural_data)
    # run by layer so that we can store individual scores
    logger.info('Scoring layers')
    layer_scores = {layer_name: score_physiology_layer(model=model, layer=layer_name, activations=layer_activations,
                                                       neural_data=neural_data)
                    for layer_name, layer_activations in activations.items()}
    max_score = max(layer_scores)
    return PhysiologyScore(source_assembly=model, target_assembly=neural_data,
                           y=max_score.y, yerr=max_score.yerr, explanation=layer_scores)


@store(ignore=['activations'])
def score_physiology_layer(model, layer, activations, neural_data):
    benchmark = load_neural_benchmark(data_name=neural_data, metric_name='neural_fit')
    score = benchmark(activations)
    return PhysiologyScore(source_assembly='{}.{}'.format(model, layer), target_assembly=neural_data,
                           y=score.y, yerr=score.yerr)


def score_anatomy(model, region_layers):
    graph = model_graph(model, layers=list(region_layers.values()))
    graph = combine_graph(graph, region_layers)
    graph = cut_graph(graph, keep_nodes=relevant_regions)
    benchmark = load_neural_benchmark(data_name='ventral_stream', metric_name='edge_ratio')
    score = benchmark(graph)
    return AnatomyScore(source_assembly=model, y=score.y)


class Score(object):
    def __init__(self, score_name, source_assembly, target_assembly, y, yerr, explanation=None):
        self.score_name = score_name
        self.source_assembly = source_assembly
        self.target_assembly = target_assembly
        self.type = type
        self.y = y
        self.yerr = yerr
        self.explanation = explanation

    def __repr__(self):
        return self.__class__.__name__ + "(" + ",".join(
            "{}={}".format(attr, val) for attr, val in self.__dict__.items()) + ")"


class PhysiologyScore(Score):
    def __init__(self, source_assembly, target_assembly, y, yerr, mapping=None, explanation=None):
        super().__init__(score_name='physiology', source_assembly=source_assembly, target_assembly=target_assembly,
                         y=y, yerr=yerr, explanation=explanation)
        self.mapping = mapping


class AnatomyScore(Score):
    def __init__(self, source_assembly, y, explanation=None):
        super().__init__(score_name='anatomy', source_assembly=source_assembly, target_assembly='ventral_stream',
                         y=y, yerr=0, explanation=explanation)
