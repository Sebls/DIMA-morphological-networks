"""Post-training analysis and model comparison."""

from morpho_net.analysis.pareto import is_pareto_efficient, extract_pareto_filters
from morpho_net.analysis.kernels import plot_structuring_elements, plot_pareto_elements
from morpho_net.analysis.curves import save_training_history, plot_training_curves
from morpho_net.analysis.experiment_plots import generate_experiment_plots

__all__ = [
    "is_pareto_efficient",
    "extract_pareto_filters",
    "plot_structuring_elements",
    "plot_pareto_elements",
    "save_training_history",
    "plot_training_curves",
    "generate_experiment_plots",
]
