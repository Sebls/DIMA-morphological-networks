"""Default: compile with Adam (or config optimizer) and ``model.fit``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from morpho_net.training.fit import compile_model, train_model
from morpho_net.training.procedures.base import TrainingProcedure


class StandardFitProcedure(TrainingProcedure):
    """Standard Keras training with optional extra callbacks (see :class:`AlphaSchedulerProcedure`)."""

    def extra_callbacks(self, config: dict[str, Any]) -> list[keras.callbacks.Callback]:
        return []

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
        train_cfg = config.get("training", {})
        compile_model(
            model,
            learning_rate=train_cfg.get("learning_rate", 0.01),
            loss=train_cfg.get("loss", "mse"),
            optimizer=train_cfg.get("optimizer", "adam"),
        )
        return train_model(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            config,
            output_dir=output_dir,
            extra_callbacks=self.extra_callbacks(config),
        )
