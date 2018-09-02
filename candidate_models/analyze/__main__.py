import logging
import sys

from candidate_models.analyze.export import create_latex_table, create_fixture
from candidate_models.analyze.figures import PaperFigures
from candidate_models.analyze.individual_behavior import compute_behavioral_differences
from candidate_models.analyze.stats import compute_benchmark_correlations, compute_correlations
from candidate_models.analyze.winners import best_models

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Plotting figures")
plotter = PaperFigures()
plotter()

logger.info("Computing stats")
best_models()
compute_correlations()
compute_benchmark_correlations()
compute_behavioral_differences()

logger.info("Exporting")
create_latex_table()
create_fixture()
