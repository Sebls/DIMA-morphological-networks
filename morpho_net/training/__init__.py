"""Training procedures and optimization logic."""

from morpho_net.training.callbacks import create_callbacks
from morpho_net.training.fit import compile_model, train_model
from morpho_net.training.registry import (
    get_training_procedure_class,
    list_training_procedures,
    register_training_procedure,
)
from morpho_net.training.train import run_training

__all__ = [
    "compile_model",
    "create_callbacks",
    "get_training_procedure_class",
    "list_training_procedures",
    "register_training_procedure",
    "run_training",
    "train_model",
]
