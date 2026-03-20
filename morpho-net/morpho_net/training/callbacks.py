"""Training callbacks."""

from __future__ import annotations

from pathlib import Path

import keras
import tensorflow as tf


def create_callbacks(
    checkpoint_dir: str | Path,
    loss_threshold: float = 1.0 / (255**2),
    monitor: str = "val_loss",
    mode: str = "min",
) -> list[keras.callbacks.Callback]:
    """Create training callbacks: checkpoint and early stopping."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.weights.h5"

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_weights_only=True,
        monitor=monitor,
        mode=mode,
        save_freq="epoch",
        save_best_only=True,
    )

    class EarlyStopCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if logs.get(monitor, float("inf")) < loss_threshold:
                print("\nReached the training threshold")
                self.model.stop_training = True

    return [EarlyStopCallback(), checkpoint]
