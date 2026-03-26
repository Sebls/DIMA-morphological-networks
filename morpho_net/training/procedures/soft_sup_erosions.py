"""Placeholder for SoftSupErosions-style optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from morpho_net.training.procedures.base import TrainingProcedure


class SoftSupErosionsProcedure(TrainingProcedure):
    """Reserved for smooth / relaxed supremum-of-erosions training objectives."""

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
            "training.update_method 'soft_sup_erosions' is not implemented yet. "
            "Implement TrainingProcedure.run (e.g. custom loss + train_step) and register it."
        )
