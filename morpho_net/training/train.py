"""High-level training entry: dispatch on config ``training.update_method``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from morpho_net.training.fit import compile_model, train_model
from morpho_net.training.registry import get_training_procedure_class


def run_training(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> keras.callbacks.History:
    """Run the training procedure selected by ``config['training']['update_method']``."""
    train_cfg = config.get("training", {})
    name = train_cfg.get("update_method") or train_cfg.get("method") or "standard_fit"
    procedure_cls = get_training_procedure_class(name)
    return procedure_cls().run(model, x_train, y_train, x_val, y_val, config, output_dir)


__all__ = ["compile_model", "train_model", "run_training"]
