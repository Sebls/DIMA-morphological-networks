"""Structured two-phase training: global loss gradient + auxiliary submodel gradient, merged per partition.

Each step: (1) forward on the full model and backprop the global loss; (2) copy weights to a
cloned auxiliary model and randomize subset A (Pareto-minimal filters, or parameters with non-zero
gradient) so subset B receives a non-sparse signal; (3) forward the auxiliary model on the same
batch and backprop the same loss; (4) merge gradients with mask 1 on A from the global step and
mask 0 on B from the auxiliary step; (5) apply with the configured optimizer.

Optional ``config['training']`` keys: ``structured_grad_eps`` (threshold for ``|g|`` in dense
partition and for non-SE weights under ``pareto``), ``secondary_random_minval`` /
``secondary_random_maxval`` (uniform range when randomizing subset A on the auxiliary model;
default from ``minval`` / ``maxval`` if present, else ``-0.45`` / ``-0.15``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import keras
import numpy as np
import tensorflow as tf

from morpho_net.analysis.pareto import is_pareto_efficient
from morpho_net.layers.dilation import MorphologicalDilation
from morpho_net.layers.erosion import MorphologicalErosion
from morpho_net.layers.sup_erosions import SupErosionsBlock, SupErosionsBlock2Inputs
from morpho_net.training.callbacks import create_callbacks, create_weight_snapshot_callback
from morpho_net.training.fit import compile_model

PartitionMode = Literal["pareto", "dense"]


def _walk_dilations(layer: keras.layers.Layer) -> list[MorphologicalDilation]:
    out: list[MorphologicalDilation] = []
    if isinstance(layer, MorphologicalDilation):
        out.append(layer)
    elif isinstance(layer, MorphologicalErosion):
        out.append(layer.dilation)
    elif isinstance(layer, SupErosionsBlock):
        out.append(layer.erosion)
    elif isinstance(layer, SupErosionsBlock2Inputs):
        pass
    else:
        for sub in getattr(layer, "layers", None) or []:
            out.extend(_walk_dilations(sub))
    return out


def collect_morphological_dilations(model: keras.Model) -> list[MorphologicalDilation]:
    """Ordered list of :class:`MorphologicalDilation` layers (structuring elements)."""
    dilations: list[MorphologicalDilation] = []
    for layer in model.layers:
        dilations.extend(_walk_dilations(layer))
    return dilations


def _dilation_var_indices(model: keras.Model) -> dict[int, MorphologicalDilation]:
    dilations = collect_morphological_dilations(model)
    idxs: list[int] = []
    for i, v in enumerate(model.trainable_variables):
        if len(v.shape) == 5 and int(v.shape[0]) == 1 and int(v.shape[1]) == 1 and int(v.shape[2]) == 1:
            idxs.append(i)
    if len(idxs) != len(dilations):
        raise ValueError(
            f"Could not align dilation weights: {len(idxs)} 5D trainable tensors vs "
            f"{len(dilations)} morphological dilation layers."
        )
    return {idxs[j]: dilations[j] for j in range(len(dilations))}


def _pareto_mask_for_weights(w_np: np.ndarray) -> np.ndarray:
    """Boolean mask (n_filters,) True = Pareto-minimal (efficient) structuring elements."""
    flat = w_np[0, 0, 0, :, :]
    filters_t = flat.T
    return is_pareto_efficient(filters_t.copy(), return_mask=True)


def _randomize_subset_a(
    w_np: np.ndarray,
    mask_a: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
) -> np.ndarray:
    """Replace entries where ``mask_a`` is True with uniform random values in ``[low, high]``."""
    out = w_np.copy()
    noise = rng.uniform(low, high, size=w_np.shape).astype(np.float32)
    m = np.broadcast_to(mask_a, w_np.shape)
    out = np.where(m, noise, out)
    return out


def _broadcast_filter_mask(mask_f: np.ndarray, w_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast (F,) filter mask to full dilation weight shape (1,1,1,P,F)."""
    m = mask_f.reshape((1, 1, 1, 1, -1))
    return np.broadcast_to(m, w_shape)


def _prepare_aux_weights(
    main: keras.Model,
    aux: keras.Model,
    partition: PartitionMode,
    g1: list[tf.Tensor | None],
    grad_eps: float,
    rng: np.random.Generator,
    rand_low: float,
    rand_high: float,
) -> None:
    """Copy main weights to aux and randomize subset A so subset B receives gradient in the auxiliary forward."""
    aux.set_weights(main.get_weights())
    dil_map = _dilation_var_indices(main)
    trainable = list(main.trainable_variables)

    for i, v in enumerate(trainable):
        w_np = v.numpy()
        if i in dil_map:
            if partition == "pareto":
                mask_f = _pareto_mask_for_weights(w_np)
            else:
                g = g1[i]
                if g is None:
                    mask_a_np = np.zeros(w_np.shape, dtype=bool)
                else:
                    mask_a_np = np.abs(g.numpy()) > grad_eps
            mask_a_full = (
                _broadcast_filter_mask(mask_f, w_np.shape)
                if partition == "pareto"
                else mask_a_np
            )
            new_w = _randomize_subset_a(w_np, mask_a_full, rng, rand_low, rand_high)
            aux.trainable_variables[i].assign(new_w)
        else:
            if partition == "pareto":
                g = g1[i]
                if g is None:
                    mask_a_np = np.zeros_like(w_np, dtype=bool)
                else:
                    mask_a_np = np.abs(g.numpy()) > grad_eps
            else:
                g = g1[i]
                if g is None:
                    mask_a_np = np.zeros_like(w_np, dtype=bool)
                else:
                    mask_a_np = np.abs(g.numpy()) > grad_eps
            new_w = _randomize_subset_a(w_np, mask_a_np, rng, rand_low, rand_high)
            aux.trainable_variables[i].assign(new_w)


def _merge_masks_for_partition(
    partition: PartitionMode,
    model: keras.Model,
    g1: list[tf.Tensor | None],
    grad_eps: float,
) -> list[np.ndarray]:
    """Per trainable variable: float mask m with 1 on subset A (use global gradient), 0 on B (use auxiliary)."""
    dil_map = _dilation_var_indices(model)
    masks: list[np.ndarray] = []
    for i, v in enumerate(model.trainable_variables):
        w_shape = tuple(v.shape)
        if i in dil_map:
            w_np = v.numpy()
            if partition == "pareto":
                mask_f = _pareto_mask_for_weights(w_np)
                m = _broadcast_filter_mask(mask_f, w_shape)
            else:
                g = g1[i]
                if g is None:
                    m = np.zeros(w_shape, dtype=np.float32)
                else:
                    m = (np.abs(g.numpy()) > grad_eps).astype(np.float32)
            masks.append(m)
        else:
            g = g1[i]
            if g is None:
                m = np.zeros(w_shape, dtype=np.float32)
            else:
                m = (np.abs(g.numpy()) > grad_eps).astype(np.float32)
            masks.append(m)
    return masks


def _merge_gradients(
    g1: list[tf.Tensor | None],
    g2: list[tf.Tensor | None],
    masks_a: list[np.ndarray],
    variables: list[tf.Variable],
) -> list[tf.Tensor]:
    merged: list[tf.Tensor] = []
    for i, v in enumerate(variables):
        ma = tf.constant(masks_a[i], dtype=v.dtype)
        mb = tf.constant(1.0, dtype=v.dtype) - ma
        g1t = g1[i] if g1[i] is not None else tf.zeros_like(v)
        g2t = g2[i] if g2[i] is not None else tf.zeros_like(v)
        g2t = tf.cast(g2t, g1t.dtype)
        merged.append(g1t * ma + g2t * mb)
    return merged


def run_structured_update_training(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    output_dir: str | Path | None,
    partition: PartitionMode,
) -> keras.callbacks.History:
    """Custom training loop: global loss gradient + auxiliary model gradient, merged by partition rule."""
    train_cfg = config.get("training", {})
    callbacks_cfg = config.get("callbacks", {})
    output_cfg = config.get("output", {})

    if output_dir is None:
        output_dir = Path(output_cfg.get("checkpoint_dir", "Checkpoint"))
    output_dir = Path(output_dir)

    compile_model(
        model,
        learning_rate=train_cfg.get("learning_rate", 0.01),
        loss=train_cfg.get("loss", "mse"),
        optimizer=train_cfg.get("optimizer", "adam"),
    )

    if y_train.ndim == 2:
        y_train = np.expand_dims(y_train, axis=-1)
    if y_val.ndim == 2:
        y_val = np.expand_dims(y_val, axis=-1)

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    batch_size = int(train_cfg.get("batch_size", 10))
    epochs = int(train_cfg.get("epochs", 2000))
    grad_eps = float(train_cfg.get("structured_grad_eps", 1e-8))
    rand_low = float(train_cfg.get("secondary_random_minval", train_cfg.get("minval", -0.45)))
    rand_high = float(train_cfg.get("secondary_random_maxval", train_cfg.get("maxval", -0.15)))
    seed = train_cfg.get("seed")
    rng = np.random.default_rng(seed)

    threshold = callbacks_cfg.get("loss_threshold", "auto")
    if threshold == "auto":
        loss_threshold = 1.0 / (255**2)
    else:
        loss_threshold = float(threshold)

    monitor = callbacks_cfg.get("checkpoint_monitor", "val_loss")
    mode = callbacks_cfg.get("checkpoint_mode", "min")

    callbacks = create_callbacks(
        checkpoint_dir=output_dir,
        loss_threshold=loss_threshold,
        monitor=monitor,
        mode=mode,
    )
    snap_k = train_cfg.get("weight_snapshot_every_k_epochs")
    snap_cb = create_weight_snapshot_callback(output_dir, snap_k)
    if snap_cb is not None:
        callbacks = list(callbacks) + [snap_cb]

    aux = keras.models.clone_model(model)
    aux.set_weights(model.get_weights())

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = model.optimizer
    if optimizer is None:
        raise RuntimeError("Model has no optimizer after compile_model.")

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    shuffle_buf = min(len(x_train), 10000)
    if shuffle_buf > 0:
        ds = ds.shuffle(shuffle_buf, seed=int(seed) if seed is not None else None, reshuffle_each_iteration=True)
    train_ds = ds

    history = keras.callbacks.History()
    history.history = {"loss": [], "val_loss": [], "mse": [], "val_mse": []}

    setattr(model, "stop_training", False)
    for cb in callbacks:
        cb.set_model(model)
        cb.on_train_begin()

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for x_batch, y_batch in train_ds:
            trainable = model.trainable_variables

            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss1 = loss_fn(y_batch, y_pred)
            g1 = tape.gradient(loss1, trainable)

            _prepare_aux_weights(
                model, aux, partition, g1, grad_eps, rng, rand_low, rand_high
            )
            masks_a = _merge_masks_for_partition(partition, model, g1, grad_eps)

            with tf.GradientTape() as tape2:
                y_pred2 = aux(x_batch, training=True)
                loss2 = loss_fn(y_batch, y_pred2)
            g2 = tape2.gradient(loss2, aux.trainable_variables)

            merged = _merge_gradients(g1, g2, masks_a, trainable)
            optimizer.apply_gradients(zip(merged, trainable))

            epoch_losses.append(float(loss1.numpy()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        y_val_p = model(x_val, training=False)
        val_loss = float(loss_fn(y_val, y_val_p).numpy())
        val_mse = float(tf.reduce_mean(tf.square(y_val - y_val_p)).numpy())

        history.history["loss"].append(train_loss)
        history.history["val_loss"].append(val_loss)
        history.history["mse"].append(train_loss)
        history.history["val_mse"].append(val_mse)

        logs = {"loss": train_loss, "val_loss": val_loss, "mse": train_loss, "val_mse": val_mse}
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs=logs)
        if getattr(model, "stop_training", False):
            break

    for cb in callbacks:
        cb.on_train_end()

    return history
