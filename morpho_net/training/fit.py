"""Compile model and run ``fit`` with checkpoint / early-stop callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from morpho_net.training.callbacks import create_callbacks, create_weight_snapshot_callback


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.01,
    loss: str = "mse",
    optimizer: str = "adam",
) -> None:
    """Compile model for training."""
    opt = keras.optimizers.Adam(learning_rate=learning_rate) if optimizer == "adam" else None
    if opt is None:
        opt = keras.optimizers.get(optimizer)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=opt,
        metrics=["mse"],
    )


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> keras.callbacks.History:
    """Train model with given data and config."""
    train_cfg = config.get("training", {})
    callbacks_cfg = config.get("callbacks", {})
    output_cfg = config.get("output", {})

    if output_dir is None:
        output_dir = Path(output_cfg.get("checkpoint_dir", "Checkpoint"))

    threshold = callbacks_cfg.get("loss_threshold", "auto")
    if threshold == "auto":
        loss_threshold = 1.0 / (255**2)
    else:
        loss_threshold = float(threshold)

    callbacks = create_callbacks(
        checkpoint_dir=output_dir,
        loss_threshold=loss_threshold,
        monitor=callbacks_cfg.get("checkpoint_monitor", "val_loss"),
        mode=callbacks_cfg.get("checkpoint_mode", "min"),
    )
    if extra_callbacks:
        callbacks = list(callbacks) + list(extra_callbacks)

    snap_k = train_cfg.get("weight_snapshot_every_k_epochs")
    snap_cb = create_weight_snapshot_callback(output_dir, snap_k)
    if snap_cb is not None:
        callbacks = list(callbacks) + [snap_cb]

    # Ensure y has correct shape for model output (N, H, W, 1)
    if y_train.ndim == 2:
        y_train = np.expand_dims(y_train, axis=-1)
    if y_val.ndim == 2:
        y_val = np.expand_dims(y_val, axis=-1)

    history = model.fit(
        x_train,
        y_train,
        batch_size=train_cfg.get("batch_size", 10),
        epochs=train_cfg.get("epochs", 2000),
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    return history
