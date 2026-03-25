"""Abstract training procedure: plug in new update rules without changing experiment runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import keras
import numpy as np


class TrainingProcedure(ABC):
    """One experiment-time training strategy (e.g. standard ``fit``, custom loops)."""

    @abstractmethod
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
        """Train ``model`` and return Keras ``History`` (or compatible object)."""
