"""Training callbacks."""

from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


def save_model_weights_npz(model: keras.Model, path: str | Path) -> None:
    """Save all layer weight tensors to a compressed ``.npz`` (keys ``layer_name__i``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, np.ndarray] = {}
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        safe = layer.name.replace("/", "_")
        for i, w in enumerate(weights):
            bundle[f"{safe}__{i}"] = np.asarray(w)
    np.savez_compressed(path, **bundle)


class WeightSnapshotCallback(keras.callbacks.Callback):
    """Save model weights every ``k`` epochs (plus initial and optional final) for analysis."""

    def __init__(
        self,
        output_dir: str | Path,
        every_k_epochs: int,
        save_initial: bool = True,
        save_final_if_needed: bool = True,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / "weight_snapshots"
        self.every_k_epochs = int(every_k_epochs)
        self.save_initial = save_initial
        self.save_final_if_needed = save_final_if_needed
        self._last_epoch_index: int = -1
        self.manifest: dict = {
            "every_k_epochs": self.every_k_epochs,
            "snapshots": [],
        }

    def _write_manifest(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _append_snapshot(self, rel_file: str, epoch_index: int, completed_epochs: int) -> None:
        self.manifest["snapshots"].append(
            {
                "epoch_index": epoch_index,
                "completed_epochs": completed_epochs,
                "file": rel_file,
            }
        )
        self._write_manifest()

    def _save(self, epoch_index: int, completed_epochs: int, suffix: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"weights_{suffix}.npz"
        path = self.output_dir / fname
        save_model_weights_npz(self.model, path)
        self._append_snapshot(fname, epoch_index, completed_epochs)

    def on_train_begin(self, logs=None) -> None:
        del logs
        if self.save_initial:
            self._save(epoch_index=-1, completed_epochs=0, suffix="initial")

    def on_epoch_end(self, epoch, logs=None) -> None:
        del logs
        self._last_epoch_index = epoch
        completed = epoch + 1
        if completed % self.every_k_epochs == 0:
            self._save(epoch_index=epoch, completed_epochs=completed, suffix=f"epoch_{completed:04d}")

    def on_train_end(self, logs=None) -> None:
        del logs
        if not self.save_final_if_needed or self.every_k_epochs <= 0:
            return
        last = self._last_epoch_index
        if last < 0:
            return
        completed = last + 1
        if completed % self.every_k_epochs == 0:
            return
        # Avoid duplicate if manifest already has this completed_epochs
        existing = {s["completed_epochs"] for s in self.manifest["snapshots"]}
        if completed in existing:
            return
        self._save(epoch_index=last, completed_epochs=completed, suffix=f"epoch_{completed:04d}_final")


def create_weight_snapshot_callback(
    checkpoint_dir: str | Path,
    every_k_epochs: int | None,
) -> WeightSnapshotCallback | None:
    """Return a weight snapshot callback, or ``None`` if disabled."""
    if every_k_epochs is None or every_k_epochs <= 0:
        return None
    return WeightSnapshotCallback(checkpoint_dir, every_k_epochs=every_k_epochs)

#Early stopping if the validation loss is smaller than the threshold (we admit that the training is succesfull in this case)
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

    # Implement callback function to stop training
    # when val loss reaches e.g. LOSS_THRESHOLD = 1.5e-05
    # From Supratim Haldar
    # https://towardsdatascience.com/neural-network-with-tensorflow-how-to-stop-training-using-callback-5c8d575c18a9

    class EarlyStopCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if logs.get(monitor, float("inf")) < loss_threshold:
                print("\nReached the training threshold")
                self.model.stop_training = True

    return [EarlyStopCallback(), checkpoint]
