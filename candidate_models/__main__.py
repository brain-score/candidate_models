import argparse
import logging
import sys

from brainscore.metrics.transformations import CrossValidation
from candidate_models import score_model, Defaults
from candidate_models.models import models, infer_image_size
from candidate_models.models.implementations import Defaults as DeepModelDefaults
from candidate_models.models.implementations import model_layers

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=models.keys())
    parser.add_argument('--model_weights', type=str, default=DeepModelDefaults.weights)
    parser.add_argument('--no-model_weights', action='store_const', const=None, dest='model_weights')
    parser.add_argument('--layers', type=str, nargs='+', default=None)
    parser.add_argument('--pca', type=int, default=DeepModelDefaults.pca_components,
                        help='Number of components to reduce the flattened features to')
    parser.add_argument('--no-pca', action='store_const', const=None, dest='pca')
    parser.add_argument('--benchmark', type=str, default=Defaults.benchmark)
    parser.add_argument('--image_size', type=int, default=-1,
                        help='size of the image (same value is used for width and height). '
                             '-1 (default) to infer from model')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    print(args.model)
    if args.image_size == -1:
        args.image_size = infer_image_size(args.model)
    args.layers = args.layers or model_layers[args.model]
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3']:
        logging.getLogger(disable_logger).setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logger.info("Running with args %s", vars(args))

    logger.info('Scoring model')
    score = score_model(args.model, layers=args.layers, weights=args.model_weights,
                        pca_components=args.pca, image_size=args.image_size,
                        benchmark=args.benchmark)
    if args.benchmark == 'brain-score':
        benchmark_values = score.attrs['raw']
        for benchmark in benchmark_values['benchmark'].values:
            print(f"# {benchmark}")
            benchmark_score = benchmark_values.sel(benchmark=benchmark)
            best_value = benchmark_score.sel(aggregation='center').values.max()
            is_best_value = benchmark_score.sel(aggregation='center').values == best_value
            print("\n".join([f"{layer}: {center:.3f}+-{error:.3f} {best}" for layer, center, error, best in zip(
                benchmark_score['layer'].values.tolist(),
                benchmark_score.sel(aggregation='center').values.tolist(),
                benchmark_score.sel(aggregation='error').values.tolist(),
                [["", "[best]"][is_best] for is_best in is_best_value])]))
            print()
    if args.benchmark == 'dicarlo.Majaj2015.temporal.IT':
        from brainscore import benchmarks
        benchmark = benchmarks.load(args.benchmark)

        # ceiling
        ceiling = benchmark.ceiling
        average_ceiling = ceiling.mean('time_bin')
        center, error = average_ceiling.sel(aggregation='center'), average_ceiling.sel(aggregation='error')
        print(f"Ceiling: {center.values:.2f}+-{error.values:.2f}")
        for time_bin in ceiling['time_bin'].values:
            time_ceiling = ceiling.sel(time_bin=time_bin)
            center, error = time_ceiling.sel(aggregation='center'), time_ceiling.sel(aggregation='error')
            print(f"  > time {time_bin}: {center.values:.2f}+-{error.values:.2f}")
        # scores
        average_score = score.mean('time_slice')
        center, error = average_score.sel(aggregation='center'), average_score.sel(aggregation='error')
        print(f"model score: {center.values:.2f}+-{error.values:.2f}")
        for time_slice in score['time_slice'].values:
            region_time_score = score.sel(time_slice=time_slice)
            center, error = region_time_score.sel(aggregation='center'), region_time_score.sel(aggregation='error')
            print(f"  > time {time_slice}: {center.values:.2f}+-{error.values:.2f}")


main()
