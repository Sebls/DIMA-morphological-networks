"""Dense partition: gradient support (non-zero vs zero) with auxiliary submodel on the inactive part."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from morpho_net.training.procedures.base import TrainingProcedure
from morpho_net.training.procedures.structured_update import run_structured_update_training


class DenseUpdateProcedure(TrainingProcedure):
    """Structured two-phase training: global loss gradient where nonzero; auxiliary loss on zero-gradient region."""

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
        return run_structured_update_training(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            config,
            output_dir,
            partition="dense",
        )
