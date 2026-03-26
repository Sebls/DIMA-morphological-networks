"""Placeholder for Pareto-based weight updates (custom training step)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from morpho_net.training.procedures.base import TrainingProcedure


class ParetoUpdateProcedure(TrainingProcedure):
    """Reserved for training that incorporates Pareto structure during optimization."""

    def run(
        self,
        model: keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        config: dict[str, Any],
        output_dir: str | Path | None,
    ) -> keras.callbacks.History:
        del model, x_train, y_train, x_val, y_val, config, output_dir
        raise NotImplementedError(
            "training.update_method 'pareto_update' is not implemented yet. "
            "Subclass TrainingProcedure and implement a custom train_step or training loop, "
            "then register it in morpho_net.training.registry."
        )
