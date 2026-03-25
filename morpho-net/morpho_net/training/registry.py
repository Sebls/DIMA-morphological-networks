"""Map ``training.update_method`` (and aliases) to :class:`TrainingProcedure` classes."""

from __future__ import annotations

from typing import Type

from morpho_net.training.procedures.alpha_scheduler import AlphaSchedulerProcedure
from morpho_net.training.procedures.base import TrainingProcedure
from morpho_net.training.procedures.dense_update import DenseUpdateProcedure
from morpho_net.training.procedures.pareto_update import ParetoUpdateProcedure
from morpho_net.training.procedures.soft_sup_erosions import SoftSupErosionsProcedure
from morpho_net.training.procedures.standard_fit import StandardFitProcedure

TRAINING_REGISTRY: dict[str, Type[TrainingProcedure]] = {
    "standard_fit": StandardFitProcedure,
    "gradient_descent": StandardFitProcedure,
    "adam": StandardFitProcedure,
    "alpha_scheduler": AlphaSchedulerProcedure,
    "pareto_update": ParetoUpdateProcedure,
    "dense_update": DenseUpdateProcedure,
    "soft_sup_erosions": SoftSupErosionsProcedure,
}


def register_training_procedure(name: str, procedure_cls: Type[TrainingProcedure]) -> None:
    """Register or override a training procedure (for extensions)."""
    TRAINING_REGISTRY[name] = procedure_cls


def list_training_procedures() -> tuple[str, ...]:
    return tuple(sorted(TRAINING_REGISTRY))


def get_training_procedure_class(name: str) -> Type[TrainingProcedure]:
    key = (name or "standard_fit").strip().lower()
    cls = TRAINING_REGISTRY.get(key)
    if cls is None:
        known = ", ".join(sorted(set(TRAINING_REGISTRY)))
        raise ValueError(f"Unknown training.update_method: {name!r}. Known: {known}")
    return cls
