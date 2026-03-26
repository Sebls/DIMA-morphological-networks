"""Named training procedures (config key ``training.update_method``)."""

from morpho_net.training.procedures.alpha_scheduler import AlphaSchedulerProcedure
from morpho_net.training.procedures.base import TrainingProcedure
from morpho_net.training.procedures.dense_update import DenseUpdateProcedure
from morpho_net.training.procedures.pareto_update import ParetoUpdateProcedure
from morpho_net.training.procedures.soft_sup_erosions import SoftSupErosionsProcedure
from morpho_net.training.procedures.standard_fit import StandardFitProcedure

__all__ = [
    "AlphaSchedulerProcedure",
    "DenseUpdateProcedure",
    "ParetoUpdateProcedure",
    "SoftSupErosionsProcedure",
    "StandardFitProcedure",
    "TrainingProcedure",
]
