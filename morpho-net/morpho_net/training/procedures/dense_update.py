"""Placeholder for dense / full-batch style morphological updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from morpho_net.training.procedures.base import TrainingProcedure


class DenseUpdateProcedure(TrainingProcedure):
    """Reserved for non-minibatch or densely coupled update rules."""

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
            "training.update_method 'dense_update' is not implemented yet. "
            "Implement TrainingProcedure.run with your dense update rule and register it."
        )
