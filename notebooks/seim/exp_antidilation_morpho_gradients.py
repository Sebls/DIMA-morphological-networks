# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="4ab037c6"
# # Learning morphological gradients with antidilations and relaxed TSEI constraints
#
# This script follows the outline: mathematical background, data, targets, models (TSEI 1-block / 2-block, dual-dilation baseline, and
# supremum-of-erosions baselines), experiments, training-curve analysis, prediction comparison, and
# interpretation.

# %% colab={"base_uri": "https://localhost:8080/"} id="acQcuWaL0I-J" outputId="83fd61cc-0469-4524-84f2-7ddf4c9d10c6"
# !pip install scikit-image matplotlib

# %% [markdown] id="c555bc79"
# ## 1. Mathematical background
#
# ### 1.1 Notation and domains
#
# Let $\mathbb{N}$, $\mathbb{Z}$ and $\mathbb{R}$ denote the sets of natural numbers (including $0$),
# integers, and real numbers, respectively.
#
# Let $\Omega \subset \mathbb{Z}^2$ be a finite rectangular grid,
# $$
# \Omega = \{0,\ldots,H-1\} \times \{0,\ldots,W-1\},
# $$
# where $H,W \in \mathbb{N}$ denote the height and width.
#
# A grayscale image is a function
# $$
# f : \Omega \to \mathbb{R}.
# $$
# In practice, values are normalized in $[0,1]$.
#
# A dataset is a collection $\{f_n\}_{n=0}^{N-1}$ of $N$ images, represented as a tensor of shape
# $(N,H,W,1)$, where the last dimension corresponds to the channel.
#
# Let $W \subset \mathbb{Z}^2$ be a finite set of offsets (structuring window).
# For $x \in \mathbb{Z}^2$, the associated patch is $\{f(x+y) : y \in W\}$.
#
# For valid (non-padded) evaluation, define
# $$
# \Omega' = \{ x \in \Omega \;:\; x+y \in \Omega \;\; \forall y \in W \},
# $$
# which has size $(H_{\mathrm{out}}, W_{\mathrm{out}})$ (see §5 for concrete constructions).
#
# For each filter index $m$, let $b^\varepsilon_{m,y}, b^{\delta^*}_{m,y} \in \mathbb{R}$ be learned
# parameters indexed by $y \in W$. A scalar $K \in \mathbb{R}$ may also be learned.
#
# ### 1.2 TSEI single-block map
#
# We use a **Trainable Sup–Inf** style morphological block (TSEI) that combines an erosion-like branch
# and an **antidilation** branch, then takes a supremum over filters and an optional global scale $K$.
# For input $f$ and patch index $x$, with 2D patches $P$ over a window $W$,
#
# $$
# \varepsilon_m(f)(x) = -\max_{y \in W}\bigl(-f(x+y) - b^{\varepsilon}_{m,y}\bigr)
#        = \min_{y \in W}\bigl(f(x+y) - b^{\varepsilon}_{m,y}\bigr),
# $$
#
# $$
# \delta_m^*(f)(x) = -\max_{y \in W}\bigl(f(x+y) + b^{\delta^*}_{m,y}\bigr)
#        = \min_{y \in W}\bigl(-f(x+y) - b^{\delta^*}_{m,y}\bigr),
# $$
#
# $$
# \psi_K(f)(x) = K \cdot \max_m \min\bigl(\varepsilon_m(f)(x),\, \delta_m^*(f)(x)\bigr).
# $$
#
# Learned tensors $b^\varepsilon, b^{\delta^*}$ play the role of (flattened) structuring-element offsets
# inside each max over the patch; $K$ relaxes the effective Lipschitz scale of the map.
#
# ### 1.3 Two-block composition
#
# A deeper map stacks two TSEI blocks with independent parameters:
#
# $$
# \Psi(f) = \psi^{(2)}\!\bigl(\psi^{(1)}(f)\bigr).
# $$
#
# With valid $k\times k$ convolutional patching, each block reduces spatial size by $k-1$ along each axis.
#
#
# ### 1.4 Baseline operators
#
# Other operators considered in this work include:
#
# - Dual-dilation constructions combining parallel dilations followed by pointwise aggregation.
# - Supremum-of-erosions operators defined as maxima over families of erosion maps
#   (see §6 for implementation details).
#
# ### 1.4.1 Dual-dilation baseline
#
# A closely related baseline applies two parallel morphological dilations (erosion as $-\delta(-f)$),
# takes a per-pixel minimum across the two branches, then a supremum over filters and a scalar scale—matching
# the “MyOwn” construction in earlier notebooks.
#
# ### 1.4.2 Supremum-of-erosions baselines
#
#  - **Single-layer**: one bank of parallel erosions and $\max_m$ over filters.
#  - **Two-layer sup-erosions**: two parallel sup-erosion maps on $f$, then a combining layer.
#  - **Two-layer receptive field**: same topology with **masked** entries in the first two dilation banks.
#
# ### 1.6 Targets used for supervision (see §5 for implementations)
#
# Select with **`run_experiment(target_key)`** — keys: **`mg9`**, **`horizontal_diff_half`**,
# **`four_pixel_maxmin`** (2×2 max−min), **`internal_gradient_pair`**, **`external_gradient_pair`**
# (horizontal two-pixel formulae). Also available in code: vertical morpho gradient variant, skimage
# internal/external gradients, and top-hats (not all are wired into the registry yet).

# %% [markdown] id="21b3ce13"
# ## 2. Environment and setup

# %% colab={"base_uri": "https://localhost:8080/"} id="bJCwmwzy_K18" outputId="670d07cd-7994-4933-80a0-a290635b6019"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/", "height": 71} id="Ba1dbcGl-6aX" outputId="4363ef67-37ec-4d1c-90ee-cdc625d0d2bf"
from google.colab import files
uploaded = files.upload()  # This will prompt you to select your BSDS500 .tgz file

import os

# Get the uploaded file name
archive_name = list(uploaded.keys())[0]

# %% id="VoSdOpAJ0Evb"
# !tar -xf BSR_bsds500.tgz

# %% id="512d3217"
from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- Hyperparameters (edit here or override in `main`) ---
RANDOM_SEED = 42
USE_SYNTHETIC_DATA = False
# Colab default in `resolve_data_pack` is `/content` if unset (expects `BSR/BSDS500/data` under it).
BSDS_ROOT: str | None = os.environ.get("BSDS_ROOT")
IMG_SIZE: tuple[int, int] = (320, 320)
N_TRAIN_SYN = 48
N_VAL_SYN = 12
N_TEST_SYN = 16

# Model sizes (keep moderate for quick runs)
NUM_FILTERS_TSEI = 48
NUM_FILTERS_MYOWN = 48
N_EROSIONS_SINGLE = 64
N_EROSIONS_TL1 = 48
N_EROSIONS_TL2 = 48
N_EROSIONS_TL3 = 64

EPOCHS_MAIN = 500  # e.g. `TSEI_EPOCHS=40` for longer runs
BATCH_SIZE = 2
LEARNING_RATE = 1e-3

KERNEL_SIZE = (3, 3)

# --- Training callbacks (see markdown + cells below) ---
# Pick one strategy: ``"none"`` | ``"threshold"`` | ``"patience"`` | ``"both"`` (threshold + patience).
TRAINING_CALLBACK_MODE: str = "both"
# Fixed threshold: stop when val_loss first drops below (``None`` = not used unless mode uses it).
VAL_LOSS_EARLY_STOP_THRESHOLD: float | None = None
EARLY_STOP_MONITOR: str = "val_loss"
# Keras EarlyStopping: stop when val_loss does not improve by min_delta for patience epochs.
EARLY_STOPPING_PATIENCE: int = 25
EARLY_STOPPING_MIN_DELTA: float = 1e-5
EARLY_STOPPING_MONITOR: str = "val_loss"
EARLY_STOPPING_RESTORE_BEST_WEIGHTS: bool = True

# %% [markdown] id="acf34c99"
# ### Training callbacks (overview)
#
# Two complementary ways to stop when training has “no more useful progress”:
#
# 1. **Fixed threshold** — stop as soon as `val_loss` drops below a success level (good for hitting a target MSE).
# 2. **Patience + `min_delta`** — Keras `EarlyStopping`: wait `patience` epochs; if improvement is smaller than `min_delta`, stop (filters tiny, useless improvements). Optionally **`restore_best_weights=True`** reloads the best epoch at the end.

# %% [markdown] id="4e09ab19"
# ### 1. Fixed threshold (`EarlyStopThresholdCallback`)
#
# Stops the moment **`val_loss < VAL_LOSS_EARLY_STOP_THRESHOLD`** (or another metric via `EARLY_STOP_MONITOR`).
# Use **`TRAINING_CALLBACK_MODE = "threshold"`** (or **`"both"`** with patience). Set the threshold in the hyperparameters cell; use **`None`** to disable this branch when mode is **`patience`** or **`none`**.

# %% id="306b5c37"
import keras
from keras import Model


class EarlyStopThresholdCallback(keras.callbacks.Callback):
    """Stop when the monitored metric is below a fixed threshold."""

    def __init__(
        self,
        loss_threshold: float,
        monitor: str = "val_loss",
    ) -> None:
        super().__init__()
        self.loss_threshold = float(loss_threshold)
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None) -> None:
        del epoch
        logs = logs or {}
        if logs.get(self.monitor, float("inf")) < self.loss_threshold:
            print("\nReached the training threshold")
            self.model.stop_training = True


# %% [markdown] id="a52eabbe"
# ### Patience + `min_delta` (no more useful progress)
#
# Use **`keras.callbacks.EarlyStopping`** with:
#
# - **`patience`** — how many epochs to wait after the last improvement before stopping.
# - **`min_delta`** — an improvement counts only if it exceeds this (filters noise-level changes).
# - **`restore_best_weights=True`** — after stopping, the model weights are rolled back to the best epoch.
#
# Example: `patience=25`, `min_delta=1e-5`, `monitor="val_loss"`.

# %% id="edb72303"
def make_early_stopping_patience_callback(
    patience: int = 25,
    min_delta: float = 1e-5,
    monitor: str = "val_loss",
    *,
    restore_best_weights: bool = True,
    verbose: int = 0,
) -> keras.callbacks.EarlyStopping:
    """Keras built-in early stopping: stop when monitored metric plateaus (no meaningful improvement)."""
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        verbose=verbose,
    )


# %% [markdown] id="1c1745c0"
# ### Choosing a callback mode
#
# Set **`TRAINING_CALLBACK_MODE`** in the hyperparameters cell:
#
# | Mode | Behaviour |
# |------|-----------|
# | **`none`** | No early stopping (only checkpoint callbacks if a run directory is used). |
# | **`threshold`** | **`EarlyStopThresholdCallback`** when `VAL_LOSS_EARLY_STOP_THRESHOLD` is set. |
# | **`patience`** | **`EarlyStopping`** with `EARLY_STOPPING_PATIENCE` / `EARLY_STOPPING_MIN_DELTA`. |
# | **`both`** | Threshold callback **and** patience `EarlyStopping` together. |
#
# You can override per run via `train_and_eval_suite(..., callback_mode=...)`.

# %% id="52bc92f5"
class SaveLastWeightsCallback(keras.callbacks.Callback):
    """Write final epoch weights once at end of training (overwrites same file)."""

    def __init__(self, filepath: str | Path) -> None:
        super().__init__()
        self.filepath = Path(filepath)

    def on_train_end(self, logs=None) -> None:
        del logs
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.filepath))


def build_training_callbacks(
    ck_root: Path | None,
    model_slug: str,
    *,
    mode: str = "none",
    val_loss_threshold: float | None = None,
    threshold_monitor: str = "val_loss",
    patience: int = 25,
    min_delta: float = 1e-5,
    patience_monitor: str = "val_loss",
    restore_best_weights: bool = True,
) -> list[keras.callbacks.Callback]:
    """Best/last checkpoints plus optional threshold and/or Keras EarlyStopping (see ``TRAINING_CALLBACK_MODE``)."""
    cbs: list[keras.callbacks.Callback] = []
    if ck_root is not None:
        d = ck_root / model_slug
        d.mkdir(parents=True, exist_ok=True)
        best = d / "best.weights.h5"
        last = d / "last.weights.h5"
        cbs.extend(
            [
                keras.callbacks.ModelCheckpoint(
                    filepath=str(best),
                    save_weights_only=True,
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                ),
                SaveLastWeightsCallback(last),
            ]
        )

    m = mode.lower().strip()
    if m in ("threshold", "both") and val_loss_threshold is not None:
        cbs.append(
            EarlyStopThresholdCallback(
                val_loss_threshold, monitor=threshold_monitor
            )
        )
    if m in ("patience", "both"):
        cbs.append(
            make_early_stopping_patience_callback(
                patience=patience,
                min_delta=min_delta,
                monitor=patience_monitor,
                restore_best_weights=restore_best_weights,
            )
        )
    return cbs


def load_best_weights_if_present(model: keras.Model, ck_root: Path, model_slug: str) -> bool:
    p = ck_root / model_slug / "best.weights.h5"
    if not p.is_file():
        return False
    model.load_weights(str(p))
    return True


def new_dated_experiment_dir(parent: Path, prefix: str = "mg9") -> Path:
    """``parent / {prefix}_{UTC date}_{time}`` with ``plots/`` and ``checkpoints/`` subdirs."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = parent / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def histories_to_json_safe(histories: dict) -> dict:
    out: dict = {}
    for name, hist in histories.items():
        h = hist.history if hasattr(hist, "history") else hist
        out[name] = {
            k: [float(x) for x in v] for k, v in h.items() if isinstance(v, list)
        }
    return out


try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is required") from e

try:
    from skimage import io as skio
    from skimage import color as skcolor
    import scipy.io as sio
except ImportError:
    skio = None
    skcolor = None
    sio = None

rng = np.random.default_rng(RANDOM_SEED)
keras.utils.set_random_seed(RANDOM_SEED)

# %% [markdown] id="50d52892"
# ## 3–4. Dataset preparation and construction
#
# **BSDS500 (optional):** set `USE_SYNTHETIC_DATA = False` and point `BSDS_ROOT` to the folder that contains
# `BSR/BSDS500/data` (or the same layout under `BSDS500/data` after extraction). Images are converted to
# **L-channel** (LAB) grayscale, center-cropped to `IMG_SIZE`, and normalized to $[0,1]$.
# If **`groundTruth/<split>/*.mat`** is present, human **boundary** maps (first annotator) are loaded into
# `boundary_train` / `boundary_val` / `boundary_test` for contour plots next to mg9 predictions.
#
# **Synthetic fallback:** random smooth-ish images in $[0,1]$ for testing the pipeline without local data.

# %% id="503da0d8"
def center_crop_2d(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = arr.shape[:2]
    oh, ow = out_hw
    y0 = max(0, h // 2 - oh // 2)
    x0 = max(0, w // 2 - ow // 2)
    return arr[y0 : y0 + oh, x0 : x0 + ow]


def build_synthetic_pack(
    n_train: int,
    n_val: int,
    n_test: int,
    hw: tuple[int, int],
) -> dict:
    """Random images in [0,1] with a bit of spatial smoothness (3x3 average)."""
    h, w = hw

    def batch(n: int) -> np.ndarray:
        x = rng.uniform(0.0, 1.0, size=(n, h, w)).astype(np.float32)
        # light blur
        from numpy.lib.stride_tricks import sliding_window_view

        pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
        win = sliding_window_view(pad, (3, 3), axis=(1, 2))
        return win.mean(axis=(-2, -1)).astype(np.float32)[..., np.newaxis]

    return {
        "x_train": batch(n_train),
        "x_val": batch(n_val),
        "x_test": batch(n_test),
        "H": h,
        "W": w,
        "synthetic": True,
    }


def load_bsds_pack(
    root: Path | str,
    img_size: tuple[int, int],
    max_train: int | None = 200,
    max_val: int | None = 100,
    max_test: int | None = 200,
) -> dict | None:
    """Load BSDS L-channel crops if `root` layout matches the Colab notebook.

    ``root`` may be:
    - the **Colab content root** (e.g. ``/content``) so that ``BSR/BSDS500/data`` exists under it; or
    - the **dataset folder** itself (e.g. ``/content/BSR/BSDS500/data``) with ``images/train`` inside.
    """
    if skio is None or skcolor is None or sio is None:
        print("scikit-image / scipy not available; cannot load BSDS.")
        return None

    root = Path(root).expanduser()
    # Already pointing at .../BSDS500/data
    if (root / "images" / "train").is_dir():
        data_dir = root
    else:
        candidates = [
            root / "BSR" / "BSDS500" / "data",
            root / "BSDS500" / "data",
        ]
        data_dir = next((p for p in candidates if p.is_dir()), None)
    if data_dir is None:
        return None

    def load_split(split: str, limit: int | None) -> tuple[np.ndarray, list[str]]:
        img_dir = data_dir / "images" / split
        if not img_dir.is_dir():
            return np.zeros((0, *img_size, 1), dtype=np.float32), []
        names = sorted(
            f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))
        )
        if limit is not None:
            names = names[:limit]
        out: list[np.ndarray] = []
        for name in names:
            p = img_dir / name
            img = skio.imread(p)
            if img.ndim == 2:
                g = img.astype(np.float32) / 255.0
            else:
                lab = skcolor.rgb2lab(img)
                lch = lab[:, :, 0]
                g = (lch / 100.0).astype(np.float32)
                g = np.clip(g, 0.0, 1.0)
            g = center_crop_2d(g, img_size)
            out.append(g[..., np.newaxis])
        if not out:
            return np.zeros((0, *img_size, 1), dtype=np.float32), []
        return np.stack(out, axis=0), names

    def load_bsds_boundary_stack(split: str, names: list[str]) -> np.ndarray | None:
        """Human boundary maps from ``groundTruth/<split>/*.mat`` (first annotator), aligned with ``names``."""
        gt_dir = data_dir / "groundTruth" / split
        if not gt_dir.is_dir() or not names:
            return None
        h, w = img_size
        rows: list[np.ndarray] = []
        for name in names:
            stem = Path(name).stem
            mat_path = gt_dir / f"{stem}.mat"
            if not mat_path.is_file():
                rows.append(np.zeros((h, w, 1), dtype=np.float32))
                continue
            try:
                gt_data = sio.loadmat(mat_path)["groundTruth"]
                boundary_struct = gt_data[0, 0]
                bd = np.asarray(boundary_struct["Boundaries"].item(), dtype=np.float32)
            except (KeyError, TypeError, ValueError, IndexError):
                rows.append(np.zeros((h, w, 1), dtype=np.float32))
                continue
            bd = center_crop_2d(bd, img_size)
            rows.append(bd[..., np.newaxis])
        return np.stack(rows, axis=0)

    x_tr, names_tr = load_split("train", max_train)
    x_va, names_va = load_split("val", max_val)
    x_te, names_te = load_split("test", max_test)
    if len(x_tr) == 0:
        return None
    h, w = img_size

    b_tr = load_bsds_boundary_stack("train", names_tr)
    b_va = load_bsds_boundary_stack("val", names_va)
    b_te = load_bsds_boundary_stack("test", names_te)

    out: dict = {
        "x_train": x_tr,
        "x_val": x_va,
        "x_test": x_te,
        "H": h,
        "W": w,
        "synthetic": False,
    }
    if b_tr is not None:
        out["boundary_train"] = b_tr
    if b_va is not None:
        out["boundary_val"] = b_va
    if b_te is not None:
        out["boundary_test"] = b_te
    if b_tr is not None or b_va is not None or b_te is not None:
        out["has_bsds_boundaries"] = True
    return out


def resolve_data_pack() -> dict:
    if USE_SYNTHETIC_DATA:
        print("Using synthetic data pack.")
        return build_synthetic_pack(N_TRAIN_SYN, N_VAL_SYN, N_TEST_SYN, IMG_SIZE)
    # Colab: put BSDS under /content/BSR/BSDS500/data → set BSDS_ROOT="/content" or pass dataset path.
    root = Path(BSDS_ROOT).expanduser() if BSDS_ROOT else Path("/content")
    pack = load_bsds_pack(root, IMG_SIZE)
    if pack is None:
        print("BSDS not found — falling back to synthetic.")
        return build_synthetic_pack(N_TRAIN_SYN, N_VAL_SYN, N_TEST_SYN, IMG_SIZE)
    return pack


# %% [markdown] id="39201c29"
# ## 5. Target definitions and generation
#
# Supervision maps used below are implemented in the following cells. Each cell pairs a short LaTeX
# definition with the corresponding NumPy builder.

# %% [markdown] id="tgt-md-horizontal-diff"
# ### Horizontal two-pixel map (half-difference)
#
# Non-increasing two-pixel map on horizontal neighbors:
#
# $$
# y(x) = \tfrac{1}{2}\bigl(f(x_{\mathrm{left}}) - f(x_{\mathrm{right}})\bigr)
# $$
#
# Output spatial size is $(H,\,W-1)$ before alignment to the TSEI valid grid.

# %% id="tgt-code-horizontal-diff"
def np_target_horizontal_diff_half(x_bch: np.ndarray) -> np.ndarray:
    """Non-increasing two-pixel map (x_left - x_right)/2; shape (N, H, W-1, 1)."""
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    return (0.5 * (x[:, :, :-1] - x[:, :, 1:]))[..., np.newaxis].astype(np.float32)


# %% [markdown] id="tgt-md-mg9"
# ### Vertical max–min gradient (“mg9” stencil)
#
# On each $3\times 3$ window position in `kernel_fi`, take vertically aligned pixels $u,v$ in the same column:
#
# $$
# y(x) = \max(u,v) - \min(u,v) = |u-v|.
# $$

# %% id="tgt-code-mg9"
def kernel_fi_mg9(x_train: np.ndarray) -> np.ndarray:
    """Vertical |u-v| gradient on the 3x3 sliding grid; output (N, H-2, W-2, 1)."""
    outputs: list[float] = []
    for sample_idx in range(x_train.shape[0]):
        x_sample = x_train[sample_idx]
        for i in range(x_train.shape[1] - 2):
            for j in range(x_train.shape[2] - 2):
                x_1 = x_sample[i, j + 1]
                x_2 = x_sample[i + 1, j + 1]
                outputs.append(max(x_1, x_2) - min(x_1, x_2))
    out = np.asarray(outputs, dtype=np.float32)
    nh, nw = x_train.shape[1] - 2, x_train.shape[2] - 2
    return out.reshape(x_train.shape[0], nh, nw, 1)


# %% [markdown] id="tgt-md-vertical-morph"
# ### Beucher-style vertical neighbor gradient (alternative stencil)
#
# On the inner $(H-1)\times(W-2)$ region, compare vertically stacked pixels $u,v$ at aligned columns:
#
# $$
# y = \max(u,v) - \min(u,v).
# $$
#
# (Different footprint than `kernel_fi_mg9`; useful as an alternative target.)

# %% id="tgt-code-vertical-morph"
def np_target_vertical_morph_gradient(x_bch: np.ndarray) -> np.ndarray:
    """Beucher-style vertical neighbor gradient on inner (H-1)x(W-2) (alternative stencil)."""
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    n, h, w = x.shape
    out = np.zeros((n, h - 1, w - 2, 1), dtype=np.float32)
    for i in range(n):
        u = x[i, :-1, 1:-1]
        v = x[i, 1:, 1:-1]
        out[i, :, :, 0] = np.maximum(u, v) - np.minimum(u, v)
    return out


# %% [markdown] id="tgt-md-disk-grad"
# ### Internal / external morphological gradient (disk $B$, skimage)
#
# Classical grayscale gradients with a planar structuring element $B$ (here `disk(1)`):
#
# $$
# \rho^-_B(f) = f - f \ominus B,\qquad \rho^+_B(f) = f \oplus B - f.
# $$

# %% id="tgt-code-disk-grad"
def np_batch_internal_gradient(x_bch: np.ndarray, fp=None) -> np.ndarray:
    from skimage.morphology import disk, erosion

    if fp is None:
        fp = disk(1)
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    y = np.stack([x[n] - erosion(x[n], fp) for n in range(x.shape[0])])
    return y[..., np.newaxis].astype(np.float32)


def np_batch_external_gradient(x_bch: np.ndarray, fp=None) -> np.ndarray:
    from skimage.morphology import disk, dilation

    if fp is None:
        fp = disk(1)
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    y = np.stack([dilation(x[n], fp) - x[n] for n in range(x.shape[0])])
    return y[..., np.newaxis].astype(np.float32)


# %% [markdown] id="tgt-md-tophats"
# ### White / black top-hats (disk $B$, skimage)
#
# $$
# \mathrm{WTH}_B(f) = f - f \circ B,\qquad \mathrm{BTH}_B(f) = f \bullet B - f.
# $$

# %% id="tgt-code-tophats"
def np_batch_white_tophat(x_bch: np.ndarray, fp=None) -> np.ndarray:
    from skimage.morphology import disk, white_tophat

    if fp is None:
        fp = disk(1)
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    y = np.stack([white_tophat(x[n], fp) for n in range(x.shape[0])])
    return y[..., np.newaxis].astype(np.float32)


def np_batch_black_tophat(x_bch: np.ndarray, fp=None) -> np.ndarray:
    from skimage.morphology import disk, black_tophat

    if fp is None:
        fp = disk(1)
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    y = np.stack([black_tophat(x[n], fp) for n in range(x.shape[0])])
    return y[..., np.newaxis].astype(np.float32)


# %% [markdown] id="tgt-md-tsei-grid"
# ### Alignment with the TSEI valid convolution grid
#
# A single TSEI block with kernel $(k_h,k_w)$ uses valid patching, so spatial size shrinks by $k_h-1$
# and $k_w-1$. After $n$ stacked blocks (same kernel),
#
# $$
# H_{\mathrm{out}} = H - n(k_h-1),\qquad W_{\mathrm{out}} = W - n(k_w-1).
# $$
#
# Targets are **top-left cropped** to the same $(H_{\mathrm{out}},W_{\mathrm{out}})$ as the supervised map.

# %% id="tgt-code-tsei-grid"
def tsei_stacked_output_shape(
    h: int, w: int, kernel_size: tuple[int, int], n_blocks: int
) -> tuple[int, int]:
    kh, kw = int(kernel_size[0]), int(kernel_size[1])
    hh, ww = int(h), int(w)
    for _ in range(int(n_blocks)):
        hh = hh - kh + 1
        ww = ww - kw + 1
    return hh, ww


def crop_y_to_output(y_bch: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    h_out = min(h_out, y_bch.shape[1])
    w_out = min(w_out, y_bch.shape[2])
    return y_bch[:, :h_out, :w_out, :]


def labels_for_n_tsei_blocks(
    y_ref: np.ndarray,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    n_blocks: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    ho, wo = tsei_stacked_output_shape(h_in, w_in, kernel_size, n_blocks)
    return crop_y_to_output(y_ref, ho, wo), (ho, wo)


def crop_target_to_tsei_grid(
    y: np.ndarray,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    n_blocks: int = 1,
) -> np.ndarray:
    """Top-left crop of target maps to the valid TSEI output grid after ``n_blocks`` (default: 1-block)."""
    ho, wo = tsei_stacked_output_shape(h_in, w_in, kernel_size, n_blocks)
    return crop_y_to_output(y, ho, wo)


# %% [markdown] id="tgt-md-four-pixel"
# ### Four-pixel morphological gradient on $2\times 2$ patches
#
# For each $2\times 2$ patch with values $x_1,\ldots,x_4$:
#
# $$
# y = \max_i x_i - \min_i x_i .
# $$

# %% id="tgt-code-four-pixel"
def np_target_four_pixel_maxmin(x_bch: np.ndarray) -> np.ndarray:
    """Four-pixel morphological gradient on each 2×2 patch: max − min; shape (N, H−1, W−1, 1)."""
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    a = x[:, :-1, :-1]
    b = x[:, :-1, 1:]
    c = x[:, 1:, :-1]
    d = x[:, 1:, 1:]
    mx = np.maximum(np.maximum(a, b), np.maximum(c, d))
    mn = np.minimum(np.minimum(a, b), np.minimum(c, d))
    return (mx - mn)[..., np.newaxis].astype(np.float32)


# %% [markdown] id="tgt-md-pair-grad"
# ### Two-pixel internal / external gradients (horizontal neighbors)
#
# Let $x_1$ be the left pixel and $x_2$ the right pixel in each horizontal pair.
#
# **Internal gradient**
# $$
# m(x_1,x_2) = x_1 - \min(x_1,x_2).
# $$
#
# **External gradient**
# $$
# m(x_1,x_2) = \max(x_1,x_2) - x_1.
# $$

# %% id="tgt-code-pair-grad"
def np_target_internal_gradient_pair(x_bch: np.ndarray) -> np.ndarray:
    """Two-pixel internal gradient: m(x₁,x₂) = x₁ − min(x₁,x₂) on horizontal neighbors; (N, H, W−1, 1)."""
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    left = x[:, :, :-1]
    right = x[:, :, 1:]
    return (left - np.minimum(left, right))[..., np.newaxis].astype(np.float32)


def np_target_external_gradient_pair(x_bch: np.ndarray) -> np.ndarray:
    """Two-pixel external gradient: m(x₁,x₂) = max(x₁,x₂) − x₁ on horizontal neighbors; (N, H, W−1, 1)."""
    x = x_bch[..., 0] if x_bch.ndim == 4 else x_bch
    left = x[:, :, :-1]
    right = x[:, :, 1:]
    return (np.maximum(left, right) - left)[..., np.newaxis].astype(np.float32)


# %% [markdown] id="tgt-md-registry"
# ### Target registry for `run_experiment(target_key=...)`
#
# String keys select which map above is used for training; all selected targets are cropped to the **one-block**
# TSEI output grid via `crop_target_to_tsei_grid`. (Morphology helpers in skimage cells are available in code but
# not every key is wired into the registry.)

# %% id="tgt-code-registry"
#: Short names for `run_experiment(target_key=...)` — all are cropped to the 1-block TSEI valid grid.
TARGET_LABELS: dict[str, str] = {
    "mg9": "mg9 (vertical |u−v| on 3×3 stencil)",
    "horizontal_diff_half": "(x_left − x_right) / 2",
    "four_pixel_maxmin": "max(2×2) − min(2×2)",
    "internal_gradient_pair": "x₁ − min(x₁,x₂) (horizontal pair)",
    "external_gradient_pair": "max(x₁,x₂) − x₁ (horizontal pair)",
}

TARGET_REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mg9": lambda x: kernel_fi_mg9(x[..., 0]),
    "horizontal_diff_half": np_target_horizontal_diff_half,
    "four_pixel_maxmin": np_target_four_pixel_maxmin,
    "internal_gradient_pair": np_target_internal_gradient_pair,
    "external_gradient_pair": np_target_external_gradient_pair,
}


def list_target_keys() -> list[str]:
    return sorted(TARGET_LABELS.keys())


def compute_targets_for_experiment(
    pack: dict,
    target_key: str,
    *,
    kernel_size: tuple[int, int] = KERNEL_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute train/val supervision tensors aligned to the 1-block TSEI output size."""
    h_in = int(pack["H"])
    w_in = int(pack["W"])
    x_tr = pack["x_train"]
    x_va = pack["x_val"]
    key = target_key.lower().strip()
    if key not in TARGET_LABELS:
        raise KeyError(
            f"Unknown target_key {target_key!r}. Use one of: {list_target_keys()}"
        )
    target_fn = TARGET_REGISTRY[key]
    y_tr = target_fn(x_tr)
    y_va = target_fn(x_va)
    y_tr = crop_target_to_tsei_grid(y_tr, h_in, w_in, kernel_size, 1)
    y_va = crop_target_to_tsei_grid(y_va, h_in, w_in, kernel_size, 1)
    meta = {
        "target_key": key,
        "target_label": TARGET_LABELS[key],
        "y_train_shape": tuple(int(x) for x in y_tr.shape),
        "y_val_shape": tuple(int(x) for x in y_va.shape),
    }
    return y_tr, y_va, meta


def target_slug_for_files(target_key: str) -> str:
    """Safe fragment for PNG filenames."""
    s = target_key.lower().strip().replace(" ", "_")
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


# %% [markdown] id="be72fb64"
# ## 6. Model definitions — TSEI block, dual-dilation baseline

# %% [markdown] id="mubzk98Z2KNm"
# Below: learnable **dilation** (building block for grayscale erosions), **TSEI** and **ScaledOutput**,
# then **supremum-of-erosions** architectures matching the usual morphological-network zoo.

# %% [markdown] id="M_engakS2Naq"
# ### 6.1 Learnable morphological dilation
#
# Discrete **dilation** with a flat structuring element $b$ on patches indexed by $y$:
#
# $$
# (f \oplus b)(x) = \max_{y \in W} \bigl( f(x+y) + b_y \bigr).
# $$
#
# Here each output **filter** $m$ has its own learnable offsets $b_{m,y}$ (stored as a vector of length
# $|W| = k_h k_w$). **Erosion** is obtained by the standard dual: $f \ominus b = -((-f) \oplus b)$.

# %% id="JQOjVbyD2Q0M"
def extract_valid_patches_expanded(
    inputs,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: str,
):
    """Extract patches and add a trailing singleton dim for per-patch offsets (shared by dilation / TSEI)."""
    patches = keras.ops.image.extract_patches(
        inputs,
        kernel_size,
        strides=stride,
        dilation_rate=1,
        padding=padding,
    )
    return keras.ops.expand_dims(patches, axis=-1)


class MorphologicalDilation(keras.layers.Layer):
    """Learnable dilation: max over patch of (f + w); weight shape (1,1,1,|W|,filters)."""

    def __init__(
        self,
        filters: int = 32,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: str = "VALID",
        minval: float = -0.45,
        maxval: float = -0.15,
        seed: int | None = None,
        kernel_initializer: keras.initializers.Initializer | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape: tuple) -> None:
        k_h, k_w = self.kernel_size
        patch_size = k_h * k_w
        if self.kernel_initializer is not None:
            initializer: keras.initializers.Initializer = self.kernel_initializer
        else:
            initializer = keras.initializers.RandomUniform(
                minval=self.minval,
                maxval=self.maxval,
                seed=self.seed,
            )
        self.w = self.add_weight(
            shape=(1, 1, 1, patch_size, self.filters),
            initializer=initializer,
            trainable=True,
            name="dilation_weights",
        )
        super().build(input_shape)

    def call(self, inputs):
        patches = extract_valid_patches_expanded(
            inputs, self.kernel_size, self.stride, self.padding
        )
        return keras.ops.max(patches + self.w, axis=3)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "minval": self.minval,
                "maxval": self.maxval,
                "seed": self.seed,
                "kernel_initializer": keras.saving.serialize_keras_object(
                    self.kernel_initializer
                )
                if self.kernel_initializer is not None
                else None,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> MorphologicalDilation:
        config = dict(config)
        raw = config.pop("kernel_initializer", None)
        if raw is not None:
            config["kernel_initializer"] = keras.saving.deserialize_keras_object(raw)
        return cls(**config)


# %% [markdown] id="NqJ7TKoK2VwD"
# ### 6.2 Two-input combining layer (second stage of two-layer sup-erosion models)
#
# Given two single-channel maps $u(x)$, $v(x)$ (same spatial size) and learned scalars $w^{(1)}_m$,
# $w^{(2)}_m$ per branch and filter $m$, the layer forms a **parallel min** then **supremum over $m$**:
#
# $$
# y(x) = \max_m \min\bigl( u(x) - w^{(1)}_m,\; v(x) - w^{(2)}_m \bigr).
# $$
#
# Components: **per-filter shifts** $w^{(k)}_m$ (broadcast over space), **elementwise min** across the two
# shifted maps for each $m$, then **max over the filter index** $m$.

# %% id="TFsqLFrJ2VL2"
class SupErosionsBlock2Inputs(keras.layers.Layer):
    """Combine two (N,H,W,1) maps: max_m min(u - w1_m, v - w2_m)."""

    def __init__(
        self,
        n_erosions: int,
        minval: float = -0.45,
        maxval: float = -0.15,
        seed: int | None = None,
        weight_initializer: keras.initializers.Initializer | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_erosions = n_erosions
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.weight_initializer = weight_initializer

    def build(self, input_shape) -> None:
        if self.weight_initializer is not None:
            initializer: keras.initializers.Initializer = self.weight_initializer
        else:
            initializer = keras.initializers.RandomUniform(
                minval=self.minval,
                maxval=self.maxval,
                seed=self.seed,
            )
        self.w1 = self.add_weight(
            shape=(self.n_erosions,),
            initializer=initializer,
            trainable=True,
            name="w1",
        )
        self.w2 = self.add_weight(
            shape=(self.n_erosions,),
            initializer=initializer,
            trainable=True,
            name="w2",
        )
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            input1, input2 = inputs
        else:
            raise ValueError("SupErosionsBlock2Inputs expects [input1, input2]")
        w1b = keras.ops.reshape(self.w1, (1, 1, 1, -1))
        w2b = keras.ops.reshape(self.w2, (1, 1, 1, -1))
        z1 = input1 - w1b
        z2 = input2 - w2b
        t = keras.ops.minimum(z1, z2)
        return keras.ops.max(t, axis=-1, keepdims=True)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "n_erosions": self.n_erosions,
                "minval": self.minval,
                "maxval": self.maxval,
                "seed": self.seed,
                "weight_initializer": keras.saving.serialize_keras_object(
                    self.weight_initializer
                )
                if self.weight_initializer is not None
                else None,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> SupErosionsBlock2Inputs:
        config = dict(config)
        raw = config.pop("weight_initializer", None)
        if raw is not None:
            config["weight_initializer"] = keras.saving.deserialize_keras_object(raw)
        return cls(**config)


# %% [markdown] id="GmMonS9m2ZXh"
# ### 6.3 Learnable output scale (`ScaledOutput`)
#
# A single trainable scalar $s$ rescales the whole map (used after the outer sup in the dual-dilation baseline):
#
# $$
# y(x) = s \cdot f(x).
# $$

# %% id="YTdNKpw52X_h"
class ScaledOutput(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(1.0),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale


# %% [markdown] id="UetFmsEk2cDJ"
# ### 6.4 TSEI morphological block (`TSEIMorphologicalBlock`)
#
# For each filter $m$, two parallel **max-pool**-style branches over a $k\times k$ patch (learned offsets
# $b^\varepsilon_m$, $b^{\delta^*}_m$), then **min** across branches, then **sup over $m$**, optional **$K$**:
#
# $$
# \varepsilon_m(f)(x) = \min_{y\in W} \bigl( f(x+y) - b^\varepsilon_{m,y} \bigr),\quad
# \delta^*_m(f)(x) = \min_{y\in W} \bigl( -f(x+y) - b^{\delta^*}_{m,y} \bigr),
# $$
#
# $$
# \psi_K(f)(x) = K \cdot \max_m \min\bigl( \varepsilon_m(f)(x),\, \delta^*_m(f)(x) \bigr).
# $$
#
# **Components:** patch extraction, **two** offset tensors (erosion-like vs antidilation-like), **min** across
# the two branches per $m$, **max over $m$**, optional **$K$**.

# %% id="MZq02GfD2dmm"
class TSEIMorphologicalBlock(keras.layers.Layer):
    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: str = "VALID",
        learnable_scale: bool = True,
        weight_min: float = -0.15,
        weight_max: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_filters = int(num_filters)
        self.kernel_size = (
            (int(kernel_size[0]), int(kernel_size[1]))
            if isinstance(kernel_size, (tuple, list))
            else (int(kernel_size), int(kernel_size))
        )
        self.stride = (
            (int(stride[0]), int(stride[1]))
            if isinstance(stride, (tuple, list))
            else (int(stride), int(stride))
        )
        self.padding = padding
        self.learnable_scale = learnable_scale
        self.weight_min = weight_min
        self.weight_max = weight_max

    def build(self, input_shape):
        kk = self.kernel_size[0] * self.kernel_size[1]
        w_init = keras.initializers.RandomUniform(
            minval=self.weight_min, maxval=self.weight_max
        )
        self.w_epsilon = self.add_weight(
            shape=(1, 1, 1, kk, self.num_filters),
            initializer=w_init,
            trainable=True,
            name="b_epsilon",
        )
        self.w_delta_star = self.add_weight(
            shape=(1, 1, 1, kk, self.num_filters),
            initializer=w_init,
            trainable=True,
            name="b_delta_star",
        )
        if self.learnable_scale:
            self.K = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(1.0),
                trainable=True,
                name="lipschitz_scale",
            )
        else:
            self.K = None

    def call(self, inputs):
        p = extract_valid_patches_expanded(
            inputs, self.kernel_size, self.stride, self.padding
        )
        erosion_like = -keras.ops.max(-p + self.w_epsilon, axis=3)
        antidilation_like = -keras.ops.max(p + self.w_delta_star, axis=3)
        inf_md = keras.ops.minimum(erosion_like, antidilation_like)
        sup = keras.ops.max(inf_md, axis=-1, keepdims=True)
        if self.K is not None:
            sup = sup * self.K
        return sup

    def get_structuring_element_pair(self) -> tuple[np.ndarray, np.ndarray]:
        we, wd = self.get_weights()[:2]
        return we[0, 0, 0, :, :], wd[0, 0, 0, :, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "learnable_scale": self.learnable_scale,
                "weight_min": self.weight_min,
                "weight_max": self.weight_max,
            }
        )
        return cfg


# %% [markdown] id="Zl9ACAUG2f-5"
# ### 6.5 TSEI and dual-dilation baseline models
#
# **Single-block TSEI:** one map $\psi_K(f)$. **Two-block:** $\psi^{(2)}(\psi^{(1)}(f))$ with independent parameters.
#
# **MyOwn baseline:** two banks of dilations on $f$ and $-f$ (erosion / antidilation branches), **min** across
# branches, **sup over filters**, then **ScaledOutput**:
#
# $$
# \eta(x) = \max_m \min\bigl( (f\ominus b_m)_\text{approx}(x),\, (-f \ominus \tilde b_m)_\text{approx}(x) \bigr),\quad
# y(x) = s \cdot \eta(x).
# $$

# %% id="OQs-Fx_42im4"
def build_tsei_single_block_model(
    height: int,
    width: int,
    num_filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    stride: tuple[int, int] = (1, 1),
    padding: str = "VALID",
    learnable_scale: bool = True,
    name: str = "tsei_single_block",
) -> Model:
    inp = keras.layers.Input(shape=(height, width, 1), name="input")
    out = TSEIMorphologicalBlock(
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        learnable_scale=learnable_scale,
        name="tsei_block",
    )(inp)
    return Model(inputs=inp, outputs=out, name=name)


def build_tsei_two_block_model(
    height: int,
    width: int,
    num_filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    stride: tuple[int, int] = (1, 1),
    padding: str = "VALID",
    learnable_scale: bool = True,
    name: str = "tsei_two_block",
) -> Model:
    inp = keras.layers.Input(shape=(height, width, 1), name="input")
    x = TSEIMorphologicalBlock(
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        learnable_scale=learnable_scale,
        name="tsei_block_1",
    )(inp)
    x = TSEIMorphologicalBlock(
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        learnable_scale=learnable_scale,
        name="tsei_block_2",
    )(x)
    return Model(inputs=inp, outputs=x, name=name)


def build_myown_tsei_baseline_model(
    height: int,
    width: int,
    num_filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    minval: float = -0.15,
    maxval: float = 0.15,
    seed: int | None = None,
) -> Model:
    """Parallel erosion / antidilation banks, min, sup, ScaledOutput — uses `MorphologicalDilation`."""
    inp = keras.layers.Input(shape=(height, width, 1), name="input")
    pad = keras.layers.ZeroPadding2D(padding=(0, 0))(inp)
    out_e = -MorphologicalDilation(
        filters=num_filters,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding="VALID",
        minval=minval,
        maxval=maxval,
        seed=seed,
        name="Erosions",
    )(-pad)
    out_ad = -MorphologicalDilation(
        filters=num_filters,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding="VALID",
        minval=minval,
        maxval=maxval,
        seed=seed,
        name="Antidilations",
    )(pad)
    out_inf = keras.layers.Minimum()([out_e, out_ad])
    xout = keras.ops.max(out_inf, axis=-1, keepdims=True)
    kout = ScaledOutput(name="K_scale")(xout)
    return Model(inputs=inp, outputs=kout, name="myown_tsei_baseline")


# %% [markdown] id="OVlf-DiI2lZr"
# ### 6.6 Single-layer supremum of erosions
#
# After symmetric zero-padding so valid $3\times3$ patches cover the full image, **parallel erosions**
# $\varepsilon_m(f) = -((-f)\oplus b_m)$ and a **supremum over filters**:
#
# $$
# y(x) = \max_m \varepsilon_m(f)(x).
# $$

# %% id="wzvOQs5Z2mge"
def build_single_sup_erosions(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    n_erosions: int = 200,
    kernel_size: tuple[int, int] = (3, 3),
    minval: float = -0.45,
    maxval: float = -0.15,
    seed: int | None = None,
    kernel_initializer: keras.initializers.Initializer | None = None,
    name: str = "single_sup_erosions",
) -> Model:
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    x = keras.layers.Input(shape=input_shape, name="input")
    x_pad = keras.layers.ZeroPadding2D(padding=padding)(x)
    erosion_out = -MorphologicalDilation(
        filters=n_erosions,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding="VALID",
        minval=minval,
        maxval=maxval,
        seed=seed,
        kernel_initializer=kernel_initializer,
        name="Erosion",
    )(-x_pad)
    out = keras.ops.max(erosion_out, axis=-1, keepdims=True)
    return Model(x, out, name=name)


# %% [markdown] id="N33mSdCA2oIh"
# ### 6.7 Two-layer supremum of erosions
#
# Two **parallel** first stages on the same padded input,
# $y^{(1)}(x)=\max_{m_1}\varepsilon^{(1)}_{m_1}(f)(x)$ and
# $y^{(2)}(x)=\max_{m_2}\varepsilon^{(2)}_{m_2}(f)(x)$, then the **two-input** combiner from §6.2:
#
# $$
# y(x) = \max_{m_3} \min\bigl( y^{(1)}(x) - w^{(1)}_{m_3},\, y^{(2)}(x) - w^{(2)}_{m_3} \bigr).
# $$

# %% id="VRoEHlUQ2pi0"
def build_two_layer_sup_erosions(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    n_erosions_block1: int = 50,
    n_erosions_block2: int = 50,
    n_erosions_block3: int = 100,
    kernel_size: tuple[int, int] = (3, 3),
    init_block1: tuple[float, float] = (-0.45, -0.15),
    init_block2: tuple[float, float] = (-0.45, -0.15),
    init_block3: tuple[float, float] = (-0.45, -0.15),
    seed: int | None = None,
    kernel_initializer_block1: keras.initializers.Initializer | None = None,
    kernel_initializer_block2: keras.initializers.Initializer | None = None,
    weight_initializer_block3: keras.initializers.Initializer | None = None,
    name: str = "two_layer_sup_erosions",
) -> Model:
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    x = keras.layers.Input(shape=input_shape, name="input")
    x_pad = keras.layers.ZeroPadding2D(padding=padding)(x)
    out1 = -MorphologicalDilation(
        filters=n_erosions_block1,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding="VALID",
        minval=init_block1[0],
        maxval=init_block1[1],
        seed=seed,
        kernel_initializer=kernel_initializer_block1,
        name="Erosions1",
    )(-x_pad)
    sup1 = keras.ops.max(out1, axis=-1, keepdims=True)
    out2 = -MorphologicalDilation(
        filters=n_erosions_block2,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding="VALID",
        minval=init_block2[0],
        maxval=init_block2[1],
        seed=seed,
        kernel_initializer=kernel_initializer_block2,
        name="Erosions2",
    )(-x_pad)
    sup2 = keras.ops.max(out2, axis=-1, keepdims=True)
    out = SupErosionsBlock2Inputs(
        n_erosions=n_erosions_block3,
        minval=init_block3[0],
        maxval=init_block3[1],
        seed=seed,
        weight_initializer=weight_initializer_block3,
        name="SupErosions_3",
    )([sup1, sup2])
    return Model(x, out, name=name)


# %% [markdown] id="eaEMWZP72rII"
# ### 6.8 Two-layer receptive field (masked dilations)
#
# Same as §6.7, but after training starts, selected **patch positions** in the first two dilation banks are
# frozen to a large negative value so those filters ignore certain stencil entries (**zone specialization**).

# %% id="WFGn5Wcr2sUA"
def _apply_receptive_field_masks(
    model: Model,
    block1_inactive: list[int],
    block2_inactive: list[int],
    inactive_value: float,
) -> None:
    if block1_inactive:
        layer = model.get_layer("Erosions1")
        w = layer.get_weights()[0]
        for idx in block1_inactive:
            w[0, 0, 0, idx, :] = inactive_value
        layer.set_weights([w])
    if block2_inactive:
        layer = model.get_layer("Erosions2")
        w = layer.get_weights()[0]
        for idx in block2_inactive:
            w[0, 0, 0, idx, :] = inactive_value
        layer.set_weights([w])


def build_two_layer_receptive_field(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    n_erosions_block1: int = 500,
    n_erosions_block2: int = 500,
    n_erosions_block3: int = 700,
    kernel_size: tuple[int, int] = (3, 3),
    block1_inactive_indices: list[int] | None = None,
    block2_inactive_indices: list[int] | None = None,
    inactive_value: float = -10.0,
    init_block1: tuple[float, float] = (-0.35, 0.35),
    init_block2: tuple[float, float] = (-0.35, 0.35),
    init_block3: tuple[float, float] = (-0.35, 0.35),
    seed: int | None = None,
    kernel_initializer_block1: keras.initializers.Initializer | None = None,
    kernel_initializer_block2: keras.initializers.Initializer | None = None,
    weight_initializer_block3: keras.initializers.Initializer | None = None,
    name: str = "two_layer_receptive_field",
) -> Model:
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    block1_inactive_indices = block1_inactive_indices or []
    block2_inactive_indices = block2_inactive_indices or []
    x = keras.layers.Input(shape=input_shape, name="input")
    x_pad = keras.layers.ZeroPadding2D(padding=padding)(x)
    dil1 = MorphologicalDilation(
        filters=n_erosions_block1,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding="VALID",
        minval=init_block1[0],
        maxval=init_block1[1],
        seed=seed,
        kernel_initializer=kernel_initializer_block1,
        name="Erosions1",
    )
    out1 = -dil1(-x_pad)
    sup1 = keras.ops.max(out1, axis=-1, keepdims=True)
    dil2 = MorphologicalDilation(
        filters=n_erosions_block2,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding="VALID",
        minval=init_block2[0],
        maxval=init_block2[1],
        seed=seed,
        kernel_initializer=kernel_initializer_block2,
        name="Erosions2",
    )
    out2 = -dil2(-x_pad)
    sup2 = keras.ops.max(out2, axis=-1, keepdims=True)
    out = SupErosionsBlock2Inputs(
        n_erosions=n_erosions_block3,
        minval=init_block3[0],
        maxval=init_block3[1],
        seed=seed,
        weight_initializer=weight_initializer_block3,
        name="SupErosions_3",
    )([sup1, sup2])
    model = Model(x, out, name=name)
    model.build((None,) + input_shape)
    _apply_receptive_field_masks(
        model,
        block1_inactive_indices,
        block2_inactive_indices,
        inactive_value,
    )
    return model


# %% [markdown] id="f1781a69"
# ### Spatial alignment
#
# TSEI outputs live on a **valid** patch grid; single/two-layer **sup-erosion** models here keep full resolution
# via symmetric padding. For a fair comparison we **top-left crop** full-resolution predictions to the
# target tensor shape (reference = TSEI 1-block grid for the mg9 experiment).

# %% id="ee6e1b47"
def crop_pred_to_y(pred: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Crop prediction batch to match y_ref spatial (H,W)."""
    _, hp, wp, _ = pred.shape
    ht, wt = y_ref.shape[1], y_ref.shape[2]
    if hp == ht and wp == wt:
        return pred
    return pred[:, :ht, :wt, :].astype(np.float32)


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def predict_mse(model: keras.Model, x: np.ndarray, y_ref: np.ndarray) -> float:
    p = model.predict(x, verbose=0)
    p = crop_pred_to_y(np.asarray(p), y_ref)
    return mse_np(p, y_ref)


def mse_crop_pred_to_true(y_true, y_pred):
    """Mean squared error after top-left cropping `y_pred` to the spatial size of `y_true`.

    Needed when full-resolution sup-erosion models output $(H,W)$ maps but targets live on a valid TSEI grid.
    """
    ht = keras.ops.shape(y_true)[1]
    wt = keras.ops.shape(y_true)[2]
    y_pred_c = y_pred[:, :ht, :wt, :]
    return keras.ops.mean(keras.ops.square(y_true - y_pred_c))


def train_model(
    model: keras.Model,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    *,
    name: str,
    ck_root: Path | None,
    lr: float,
    epochs: int,
    batch_size: int,
    verbose: int,
    callbacks: list[keras.callbacks.Callback],
) -> tuple[keras.callbacks.History, float]:
    """Compile, fit, optionally reload best checkpoint, return history and validation MSE."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), loss=mse_crop_pred_to_true
    )
    history = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
    if ck_root is not None:
        load_best_weights_if_present(model, ck_root, name)
    val_mse = predict_mse(model, x_va, y_va)
    return history, val_mse


def extract_k_constants_from_model(model: keras.Model) -> dict[str, float]:
    """Learnable global scales: TSEI ``lipschitz_scale`` per block, MyOwn ``ScaledOutput``."""
    out: dict[str, float] = {}
    for layer in model.layers:
        if isinstance(layer, TSEIMorphologicalBlock) and layer.K is not None:
            k_arr = keras.ops.convert_to_numpy(layer.K)
            out[layer.name or "tsei_block"] = float(np.asarray(k_arr).reshape(()))
        elif isinstance(layer, ScaledOutput):
            s_arr = keras.ops.convert_to_numpy(layer.scale)
            out[layer.name or "K_scale"] = float(np.asarray(s_arr).reshape(()))
    return out


# %% [markdown] id="4a6d112a"
# ## 7. TSEI experiments — training suite
#
# Training uses `mse_crop_pred_to_true`: MSE after cropping the model output to the target’s height and
# width (top-left crop). That matches full-resolution sup-erosion outputs to the valid-grid targets used
# by TSEI / MyOwn.
#
# Use ``models_to_train`` to select architectures (default: TSEI 1-block, TSEI 2-block, MyOwn — no
# supremum-of-erosions baselines).

# %% id="2f61e0e2"
ALL_MODEL_SLUGS: tuple[str, ...] = (
    "tsei_1block",
    "tsei_2block",
    "myown_tsei",
    "single_sup",
    "two_layer_sup",
    "two_layer_rf",
)

#: Default experiment: main TSEI line + MyOwn only (no ``single_sup`` / ``two_layer_*``).
DEFAULT_EXPERIMENT_MODELS: tuple[str, ...] = (
    "tsei_1block",
    "tsei_2block",
    "myown_tsei",
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "tsei_1block": "TSEI 1-block",
    "tsei_2block": "TSEI 2-block",
    "myown_tsei": "MyOwn baseline",
    "single_sup": "Single sup-erosions",
    "two_layer_sup": "Two-layer sup-erosions",
    "two_layer_rf": "Two-layer receptive field",
}

#: Preferred order for “three-way” TSEI / MyOwn comparison plots (subset filtered by what was trained).
PREFERRED_THREE_WAY_ORDER: tuple[str, ...] = (
    "tsei_1block",
    "tsei_2block",
    "myown_tsei",
)


def parse_models_to_train(models: Sequence[str] | None) -> list[str]:
    """Validate and de-duplicate slugs; default :data:`DEFAULT_EXPERIMENT_MODELS`."""
    raw = list(models) if models is not None else list(DEFAULT_EXPERIMENT_MODELS)
    if not raw:
        raise ValueError("models_to_train must be a non-empty sequence")
    allowed = set(ALL_MODEL_SLUGS)
    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        key = str(s).strip()
        if key not in allowed:
            raise ValueError(
                f"Unknown model slug {s!r}. Use one of: {list(ALL_MODEL_SLUGS)}"
            )
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def comparison_plot_slugs(suite: dict) -> list[str]:
    """Slugs for overlay / three-column prediction figures, in canonical order."""
    have = suite.get("models") or {}
    return [s for s in PREFERRED_THREE_WAY_ORDER if s in have]


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "tsei_1block": {
        "tsei_blocks": 1,
        "build": lambda h, w, inp, ks: build_tsei_single_block_model(
            h, w, NUM_FILTERS_TSEI, kernel_size=ks, learnable_scale=True
        ),
    },
    "tsei_2block": {
        "tsei_blocks": 2,
        "build": lambda h, w, inp, ks: build_tsei_two_block_model(
            h, w, NUM_FILTERS_TSEI, kernel_size=ks, learnable_scale=True
        ),
    },
    "myown_tsei": {
        "tsei_blocks": 1,
        "build": lambda h, w, inp, ks: build_myown_tsei_baseline_model(
            h, w, NUM_FILTERS_MYOWN, kernel_size=ks
        ),
    },
    "single_sup": {
        "tsei_blocks": 1,
        "build": lambda h, w, inp, ks: build_single_sup_erosions(
            input_shape=inp,
            n_erosions=N_EROSIONS_SINGLE,
            kernel_size=ks,
            minval=-0.45,
            maxval=-0.15,
            seed=RANDOM_SEED,
        ),
    },
    "two_layer_sup": {
        "tsei_blocks": 1,
        "build": lambda h, w, inp, ks: build_two_layer_sup_erosions(
            input_shape=inp,
            n_erosions_block1=N_EROSIONS_TL1,
            n_erosions_block2=N_EROSIONS_TL2,
            n_erosions_block3=N_EROSIONS_TL3,
            kernel_size=ks,
        ),
    },
    "two_layer_rf": {
        "tsei_blocks": 1,
        "build": lambda h, w, inp, ks: build_two_layer_receptive_field(
            input_shape=inp,
            n_erosions_block1=N_EROSIONS_TL1,
            n_erosions_block2=N_EROSIONS_TL2,
            n_erosions_block3=N_EROSIONS_TL3,
            kernel_size=ks,
        ),
    },
}


def build_experiment_model(
    slug: str,
    *,
    h_in: int,
    w_in: int,
    input_shape: tuple[int, int, int],
    kernel_size: tuple[int, int],
) -> keras.Model:
    """Factory for experiment architectures (same mapping as training and checkpoint reload)."""
    if slug not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model slug {slug!r}")
    return MODEL_REGISTRY[slug]["build"](h_in, w_in, input_shape, kernel_size)


def train_and_eval_suite(
    pack: dict,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    models_to_train: Sequence[str] | None = None,
    kernel_size: tuple[int, int] = KERNEL_SIZE,
    epochs: int = EPOCHS_MAIN,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    verbose: int = 1,
    checkpoint_root: Path | None = None,
    callback_mode: str = TRAINING_CALLBACK_MODE,
    val_loss_threshold: float | None = VAL_LOSS_EARLY_STOP_THRESHOLD,
    threshold_monitor: str = EARLY_STOP_MONITOR,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA,
    early_stopping_monitor: str = EARLY_STOPPING_MONITOR,
    early_stopping_restore_best_weights: bool = EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
) -> dict:
    """Train only the architectures listed in ``models_to_train`` (default: TSEI 1/2 + MyOwn).

    If ``checkpoint_root`` is set, each trained model is saved under ``checkpoint_root/<name>/`` as
    ``best.weights.h5`` (lowest ``val_loss``) and ``last.weights.h5`` (final epoch only).
    Reported ``val_mse`` uses **best** weights when those files exist (after fit, including
    Keras ``EarlyStopping`` with ``restore_best_weights`` when used).

    **Callback selection:** ``callback_mode`` is one of ``"none"``, ``"threshold"``, ``"patience"``,
    ``"both"`` (see markdown “Choosing a callback mode”). Threshold mode requires
    ``val_loss_threshold``; patience mode uses ``keras.callbacks.EarlyStopping`` with patience and
    ``min_delta``.
    """
    h_in = int(pack["H"])
    w_in = int(pack["W"])
    x_tr = pack["x_train"]
    x_va = pack["x_val"]
    input_shape = (h_in, w_in, 1)

    slugs = parse_models_to_train(models_to_train)
    y_tr_2 = y_va_2 = None
    if "tsei_2block" in slugs:
        y_tr_2, _ = labels_for_n_tsei_blocks(y_train, h_in, w_in, kernel_size, 2)
        y_va_2, _ = labels_for_n_tsei_blocks(y_val, h_in, w_in, kernel_size, 2)

    ck = checkpoint_root
    results: dict = {
        "input_shape": input_shape,
        "histories": {},
        "val_mse": {},
        "models": {},
        "models_to_train": slugs,
        "checkpoint_root": str(ck) if ck is not None else None,
    }

    def _cbs(name: str) -> list[keras.callbacks.Callback]:
        return build_training_callbacks(
            ck,
            name,
            mode=callback_mode,
            val_loss_threshold=val_loss_threshold,
            threshold_monitor=threshold_monitor,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            patience_monitor=early_stopping_monitor,
            restore_best_weights=early_stopping_restore_best_weights,
        )

    for slug in slugs:
        spec = MODEL_REGISTRY[slug]
        y_tr_use, y_va_use = y_train, y_val
        if int(spec["tsei_blocks"]) == 2:
            assert y_tr_2 is not None and y_va_2 is not None
            y_tr_use, y_va_use = y_tr_2, y_va_2
        model = build_experiment_model(
            slug,
            h_in=h_in,
            w_in=w_in,
            input_shape=input_shape,
            kernel_size=kernel_size,
        )
        disp = MODEL_DISPLAY_NAMES.get(slug, slug)
        print("\n" + "=" * 72)
        print(f"Model summary — before training: {disp} ({slug})")
        print("=" * 72)
        model.summary()
        history, val_mse = train_model(
            model,
            x_tr,
            y_tr_use,
            x_va,
            y_va_use,
            name=slug,
            ck_root=ck,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=_cbs(slug),
        )
        results["histories"][slug] = history
        results["models"][slug] = model
        results["val_mse"][slug] = val_mse
        k_after = extract_k_constants_from_model(model)
        print(f"\n--- After training: {disp} ({slug}) ---")
        print(f"  Validation MSE (cropped, best checkpoint if saved): {val_mse:.6f}")
        if k_after:
            for k_name, k_val in k_after.items():
                print(f"  K ({k_name}): {k_val:.6f}")
        else:
            print("  K: (no learnable global scale in this architecture)")

    return results


# Backward-compatible alias for the full zoo of slugs.
MODEL_SLUGS: tuple[str, ...] = ALL_MODEL_SLUGS


class HistoryDict:
    """Minimal stand-in for ``keras.callbacks.History`` (``.history`` dict) for replay from ``metrics.json``."""

    __slots__ = ("history",)

    def __init__(self, history: dict) -> None:
        self.history = history


def build_all_models_untrained(
    pack: dict,
    *,
    kernel_size: tuple[int, int] = KERNEL_SIZE,
    model_slugs: Sequence[str] | None = None,
) -> dict[str, keras.Model]:
    """Instantiate a subset of architectures (default: all of :data:`ALL_MODEL_SLUGS`).

    Pass the same slugs as in training / ``metrics.json`` when reloading checkpoints for plots.
    """
    h_in = int(pack["H"])
    w_in = int(pack["W"])
    input_shape = (h_in, w_in, 1)
    slugs = list(model_slugs) if model_slugs is not None else list(ALL_MODEL_SLUGS)
    builders: dict[str, keras.Model] = {}

    def _add(slug: str) -> None:
        if slug in builders:
            return
        builders[slug] = build_experiment_model(
            slug,
            h_in=h_in,
            w_in=w_in,
            input_shape=input_shape,
            kernel_size=kernel_size,
        )

    for s in slugs:
        _add(s)
    return builders


def load_suite_for_visualization(
    run_dir: Path | str,
    *,
    target_key: str | None = None,
) -> tuple[dict, np.ndarray, dict, dict]:
    """Reload best checkpoints + histories from a finished run (``metadata.json`` + ``metrics.json``).

    Returns ``(pack, y_val, suite, run_meta)`` where ``suite`` has ``histories`` from disk and ``models``
    with **best** weights loaded. ``target_key`` defaults to the value stored in ``metadata.json``.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    metap = run_dir / "metadata.json"
    mp = run_dir / "metrics.json"
    if not metap.is_file():
        raise FileNotFoundError(f"Missing {metap}")
    if not mp.is_file():
        raise FileNotFoundError(f"Missing {mp}")
    with open(metap) as f:
        run_meta = json.load(f)
    with open(mp) as f:
        metrics = json.load(f)
    tk = (target_key or run_meta.get("target_key") or "mg9").lower().strip()
    pack = resolve_data_pack()
    _y_tr, y_va, _tm = compute_targets_for_experiment(pack, tk)
    hist_keys = list(metrics.get("histories", {}).keys())
    models = build_all_models_untrained(pack, model_slugs=hist_keys)
    ck_root = run_dir / "checkpoints"
    for slug in hist_keys:
        if slug in models:
            load_best_weights_if_present(models[slug], ck_root, slug)
    histories: dict[str, HistoryDict] = {}
    for name, hdict in metrics.get("histories", {}).items():
        histories[name] = HistoryDict(hdict)
    suite: dict = {
        "histories": histories,
        "models": models,
        "val_mse": dict(metrics.get("val_mse", {})),
        "checkpoint_root": str(ck_root),
        "experiment_run_dir": str(run_dir),
    }
    return pack, y_va, suite, run_meta


# %% [markdown] id="f060c654"
# ## 8–9. Training-curve analysis and prediction comparison

# %% id="09d51eef"

def plot_training_curves_inline(
    history: dict,
    save_path: str | Path,
    title: str,
    test_mse: float | None = None,
    *,
    log_scale: bool = False,
) -> None:
    """Plot training curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(loss) + 1)
    if loss:
        ax.plot(epochs, loss, "b-", label="Training loss", linewidth=1.5)
    if val_loss:
        ax.plot(epochs, val_loss, "r-", label="Validation loss", linewidth=1.5)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Loss (MSE) [log]", fontsize=12)
    else:
        ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title(title, fontsize=14)
    if test_mse is not None:
        ax.axhline(test_mse, color="g", linestyle="--", label=f"test_mse={test_mse:.6g}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both" if log_scale else "major")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)



def _show_fig_if_interactive() -> None:
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


_OVERLAY_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)


def plot_results_overlay(
    results: dict,
    *,
    log_scale: bool = False,
    mode: str = "val",
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    """Overlay training curves: ``mode`` is ``"val"`` or ``"train_val"`` (stacked train + val)."""
    mode_norm = mode.strip().lower()
    if mode_norm not in ("val", "train_val"):
        raise ValueError("mode must be 'val' or 'train_val'")
    default_titles = {
        ("val", False): "Validation MSE (per epoch)",
        ("val", True): "Validation loss — all models (log scale)",
        ("train_val", False): "Train vs val loss",
        ("train_val", True): "Train / val — all models (log scale)",
    }
    if title is None:
        title = default_titles[(mode_norm, log_scale)]

    colors = _OVERLAY_COLORS
    items = list(results["histories"].items())

    if mode_norm == "val":
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (name, hist) in enumerate(items):
            vl = hist.history.get("val_loss", [])
            if not vl:
                continue
            x = range(1, len(vl) + 1)
            c = colors[i % len(colors)]
            if log_scale:
                ax.semilogy(x, np.maximum(vl, 1e-12), color=c, label=name, linewidth=1.4)
            else:
                ax.plot(x, vl, color=c, label=name, linewidth=1.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("val_loss (MSE), log scale" if log_scale else "val_loss (MSE)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, which="both" if log_scale else "major")
    else:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for i, (name, hist) in enumerate(items):
            c = colors[i % len(colors)]
            loss = hist.history.get("loss", [])
            vl = hist.history.get("val_loss", [])
            if log_scale:
                if loss:
                    ax0.semilogy(
                        range(1, len(loss) + 1),
                        np.maximum(loss, 1e-12),
                        color=c,
                        label=name,
                        linewidth=1.2,
                    )
                if vl:
                    ax1.semilogy(
                        range(1, len(vl) + 1),
                        np.maximum(vl, 1e-12),
                        color=c,
                        label=name,
                        linewidth=1.2,
                    )
            else:
                if loss:
                    ax0.plot(
                        range(1, len(loss) + 1),
                        loss,
                        color=c,
                        label=name,
                        linewidth=1.2,
                    )
                if vl:
                    ax1.plot(
                        range(1, len(vl) + 1),
                        vl,
                        color=c,
                        label=name,
                        linewidth=1.2,
                    )
        ax0.set_ylabel("Training loss (log)" if log_scale else "Training loss")
        ax0.legend(loc="upper right", fontsize=7)
        ax0.grid(True, alpha=0.3, which="both" if log_scale else "major")
        ax1.set_ylabel("Validation loss (log)" if log_scale else "Validation loss")
        ax1.set_xlabel("Epoch")
        ax1.legend(loc="upper right", fontsize=7)
        ax1.grid(True, alpha=0.3, which="both" if log_scale else "major")
        fig.suptitle(title, y=1.02)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    _show_fig_if_interactive()
    plt.close(fig)


def plot_val_loss_overlay(
    results: dict,
    title: str = "Validation MSE (per epoch)",
    save_path: Path | None = None,
) -> None:
    plot_results_overlay(
        results, log_scale=False, mode="val", title=title, save_path=save_path
    )


def plot_training_overlay_train_val(
    results: dict,
    title: str = "Train vs val loss",
    save_path: Path | None = None,
) -> None:
    plot_results_overlay(
        results, log_scale=False, mode="train_val", title=title, save_path=save_path
    )


def plot_val_loss_overlay_log(
    results: dict,
    title: str = "Validation loss — all models (log scale)",
    save_path: Path | None = None,
) -> None:
    plot_results_overlay(
        results, log_scale=True, mode="val", title=title, save_path=save_path
    )


def plot_training_overlay_train_val_log(
    results: dict,
    title: str = "Train / val — all models (log scale)",
    save_path: Path | None = None,
) -> None:
    plot_results_overlay(
        results, log_scale=True, mode="train_val", title=title, save_path=save_path
    )


def visualize_tsei_pair_grids(
    tsei_layer: TSEIMorphologicalBlock,
    kh: int,
    kw: int,
    max_filters: int = 6,
    title_prefix: str = "",
    save_path: Path | None = None,
) -> None:
    be, bd = tsei_layer.get_structuring_element_pair()
    m = min(max_filters, be.shape[1])
    fig, axes = plt.subplots(2, m, figsize=(2 * m, 4), squeeze=False)
    for j in range(m):
        axes[0, j].imshow(be[:, j].reshape(kh, kw), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[0, j].set_title(f"{title_prefix}B_eps[{j}]")
        axes[0, j].axis("off")
        axes[1, j].imshow(bd[:, j].reshape(kh, kw), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[1, j].set_title(f"B_delta*[{j}]")
        axes[1, j].axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    _show_fig_if_interactive()
    plt.close(fig)


def plot_prediction_row(
    pack: dict,
    results: dict,
    y_val: np.ndarray,
    n_samples: int = 3,
    save_path: Path | None = None,
) -> None:
    """One row per sample: input crop | target | each model prediction."""
    x_va = pack["x_val"]
    n = min(n_samples, len(x_va))
    model_items = list(results["models"].items())
    ncols = 2 + len(model_items)
    fig, axes = plt.subplots(n, ncols, figsize=(2.2 * ncols, 2.4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for r in range(n):
        axes[r, 0].imshow(x_va[r, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"#{r}", rotation=0, labelpad=12)
        vmin = float(np.min(y_val[r]))
        vmax = float(np.max(y_val[r]))
        axes[r, 1].imshow(y_val[r, :, :, 0], cmap="gray", vmin=vmin, vmax=vmax)
        if r == 0:
            axes[r, 0].set_title("Input")
            axes[r, 1].set_title("Target")
        for j, (name, model) in enumerate(model_items):
            p = model.predict(x_va[r : r + 1], verbose=0)
            p = crop_pred_to_y(np.asarray(p), y_val[r : r + 1])
            span = max(vmax - vmin, 1e-6)
            axes[r, j + 2].imshow(
                p[0, :, :, 0], cmap="gray", vmin=vmin - 0.05 * span, vmax=vmax + 0.05 * span
            )
            if r == 0:
                axes[r, j + 2].set_title(name[:12], fontsize=8)
        for c in range(ncols):
            axes[r, c].axis("off")
    plt.suptitle("Validation: input | target | models (cropped preds)", y=1.02)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    _show_fig_if_interactive()
    plt.close(fig)


try:
    from morpho_net.analysis.kernels import plot_structuring_elements
except ImportError:
    plot_structuring_elements = None


def save_per_model_curves(results: dict, out_dir: Path | None) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, hist in results["histories"].items():
        hdict = hist.history
        plot_training_curves_inline(
            hdict,
            out_dir / f"{name}_loss.png",
            title=f"{name} — training",
            log_scale=False,
        )
        plot_training_curves_inline(
            hdict,
            out_dir / f"{name}_loss_log.png",
            title=f"{name} — training (log)",
            log_scale=True,
        )


def plot_training_curves_overlay(
    results: dict,
    *,
    log_scale: bool = False,
    save_path: Path | None = None,
    title: str | None = None,
    target_label: str = "target",
    model_slugs: Sequence[str] | None = None,
) -> None:
    """Overlay training and validation loss curves for a chosen subset of models (see ``model_slugs``)."""
    keys = (
        list(model_slugs)
        if model_slugs is not None
        else list(PREFERRED_THREE_WAY_ORDER)
    )
    hists: list[tuple[str, dict]] = []
    labels_short: list[str] = []
    for k in keys:
        h = results["histories"].get(k)
        if h is None:
            continue
        hi = h.history if hasattr(h, "history") else h
        hists.append((k, hi))
        labels_short.append(MODEL_DISPLAY_NAMES.get(k, k))
    if not hists:
        return
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    def _plot_series(ax, y: list, lab: str, c: str) -> None:
        x = range(1, len(y) + 1)
        if log_scale:
            ax.semilogy(x, np.maximum(y, 1e-12), color=c, label=lab, linewidth=1.5)
        else:
            ax.plot(x, y, color=c, label=lab, linewidth=1.5)

    for ax, metric_key, ylabel in (
        (ax0, "loss", "Training loss (MSE)"),
        (ax1, "val_loss", "Validation loss (MSE)"),
    ):
        for i, ((slug, hi), lab) in enumerate(zip(hists, labels_short)):
            c = colors[i % len(colors)]
            s = hi.get(metric_key, [])
            if s:
                _plot_series(ax, s, lab, c)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, which="both" if log_scale else "major")
        ax.legend(loc="upper right", fontsize=9)
    ax1.set_xlabel("Epoch", fontsize=12)
    suf = " (log scale)" if log_scale else ""
    fig.suptitle(
        title or f"{target_label} — training vs validation{suf}",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    _show_fig_if_interactive()
    plt.close(fig)


def plot_model_predictions_grid(
    pack: dict,
    results: dict,
    y_val: np.ndarray,
    *,
    aux_gt: np.ndarray | None = None,
    aux_title: str = "Auxiliary GT",
    aux_cmap: str = "gray",
    input_title: str = "Input",
    use_all_models: bool = False,
    n_samples: int = 4,
    save_path: Path | None = None,
    suptitle: str | None = None,
    target_label: str = "target",
    model_slugs: Sequence[str] | None = None,
) -> None:
    """Plot input, optional auxiliary map, supervision target, and per-model predictions.

    Use ``model_slugs`` to choose models, or leave it ``None`` to use
    :func:`comparison_plot_slugs`. Set ``use_all_models=True`` to plot every entry in
    ``results[\"models\"]`` (ignores ``model_slugs``).
    """
    models_dict = results.get("models") or {}
    if use_all_models:
        slugs = list(models_dict.keys())
    elif model_slugs is not None:
        slugs = list(model_slugs)
    else:
        slugs = comparison_plot_slugs(results)
    if not slugs:
        return
    x_va = pack["x_val"]
    has_aux = aux_gt is not None
    if has_aux:
        n = min(n_samples, len(x_va), len(aux_gt), len(y_val))
    else:
        n = min(n_samples, len(x_va), len(y_val))
    if n <= 0:
        return
    x_s = x_va[:n]
    y_s = y_val[:n]
    aux_s = aux_gt[:n] if has_aux else None
    preds: list[np.ndarray] = []
    for slug in slugs:
        m = models_dict[slug]
        p = m.predict(x_s, verbose=0)
        preds.append(crop_pred_to_y(np.asarray(p), y_s))
    ncols = (3 if has_aux else 2) + len(slugs)
    fig_w = max(14 if not has_aux else 16, 2.3 * ncols)
    fig, axes = plt.subplots(n, ncols, figsize=(fig_w, 2.8 * n), squeeze=False)
    pred_labels = [f"Pred — {MODEL_DISPLAY_NAMES.get(s, s)}" for s in slugs]
    if has_aux:
        col_titles = [input_title, aux_title, f"Target ({target_label})"] + pred_labels
    else:
        col_titles = [input_title, f"Target ({target_label})"] + pred_labels
    tgt_col = 2 if has_aux else 1
    pred0 = tgt_col + 1
    for r in range(n):
        vmin_gt = float(np.min(y_s[r]))
        vmax_gt = float(np.max(y_s[r]))
        span = max(vmax_gt - vmin_gt, 1e-6)
        vmin_p = vmin_gt - 0.05 * span
        vmax_p = vmax_gt + 0.05 * span
        axes[r, 0].imshow(x_s[r, :, :, 0], cmap="gray", vmin=0, vmax=1)
        if has_aux and aux_s is not None:
            axes[r, 1].imshow(aux_s[r, :, :, 0], cmap=aux_cmap)
        axes[r, tgt_col].imshow(y_s[r, :, :, 0], cmap="gray", vmin=vmin_gt, vmax=vmax_gt)
        for j, pred in enumerate(preds):
            axes[r, pred0 + j].imshow(pred[r, :, :, 0], cmap="gray", vmin=vmin_p, vmax=vmax_p)
        if r == 0:
            for c in range(ncols):
                axes[0, c].set_title(col_titles[c], fontsize=10)
        for c in range(ncols):
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(f"#{r}", fontsize=10, rotation=0, labelpad=16)
    default_sup = (
        f"{target_label}: input, auxiliary map, target, predictions"
        if has_aux
        else f"{target_label}: input, target, predictions"
    )
    plt.suptitle(suptitle or default_sup, fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    _show_fig_if_interactive()
    plt.close(fig)


def plot_suite_structuring_elements(
    results: dict,
    kernel_shape: tuple[int, int],
    save_dir: Path | None,
    *,
    n_show: int = 25,
    show: bool = False,
    experiment_label: str = "experiment",
) -> list[Path]:
    """Export structuring-element visualizations for models present in ``results`` (TSEI, MyOwn, etc.)."""
    _pse = plot_structuring_elements
    if _pse is None:
        return []
    written: list[Path] = []
    models = results.get("models") or {}
    specs: list[tuple[np.ndarray, str]] = []
    if "tsei_1block" in models:
        m1 = models["tsei_1block"]
        lay1 = m1.get_layer("tsei_block")
        w_eps, w_del = lay1.get_weights()[0], lay1.get_weights()[1]
        specs.extend(
            [
                (w_eps, "tsei1_b_epsilon"),
                (w_del, "tsei1_b_delta_star"),
            ]
        )
    if "tsei_2block" in models:
        m2 = models["tsei_2block"]
        for b in (1, 2):
            lay = m2.get_layer(f"tsei_block_{b}")
            we, wd = lay.get_weights()[0], lay.get_weights()[1]
            specs.append((we, f"tsei2_block{b}_b_epsilon"))
            specs.append((wd, f"tsei2_block{b}_b_delta_star"))
    if "myown_tsei" in models:
        mb = models["myown_tsei"]
        specs.append((mb.get_layer("Erosions").get_weights()[0], "baseline_erosions"))
        specs.append((mb.get_layer("Antidilations").get_weights()[0], "baseline_antidilations"))
    if "single_sup" in models:
        specs.append(
            (
                models["single_sup"].get_layer("Erosion").get_weights()[0],
                "single_sup_erosion",
            )
        )
    if not specs:
        return []

    for weights, slug in specs:
        title = f"{experiment_label} — structuring elements ({slug})"
        sp: Path | None = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            sp = save_dir / f"{slug}.png"
        paths = _pse(
            weights,
            kernel_shape=kernel_shape,
            n_show=min(n_show, 64),
            cols=5,
            filters_per_page=25,
            title=title,
            save_path=sp,
            show=show,
        )
        if paths:
            written.extend(paths)
        elif sp is not None and sp.is_file():
            written.append(sp)
    return written


def run_comparison_plots(
    suite: dict,
    pack: dict,
    y_val: np.ndarray,
    plots_dir: Path,
    *,
    target_key: str = "mg9",
    target_label: str | None = None,
    show: bool = True,
    n_pred_samples: int = 4,
) -> list[Path]:
    """Training overlays, per-model curves, GT vs prediction grids, structuring-element figures."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tkey = target_key.lower().strip()
    tlab = target_label or TARGET_LABELS.get(tkey, tkey)
    tslug = target_slug_for_files(tkey)
    saved: list[Path] = []
    comp = comparison_plot_slugs(suite)
    if comp:
        plot_training_curves_overlay(
            suite,
            log_scale=False,
            save_path=plots_dir / f"{tslug}_tsei_three_way_overlay.png",
            target_label=tlab,
            model_slugs=comp,
        )
        saved.append(plots_dir / f"{tslug}_tsei_three_way_overlay.png")
        plot_training_curves_overlay(
            suite,
            log_scale=True,
            save_path=plots_dir / f"{tslug}_tsei_three_way_overlay_log.png",
            target_label=tlab,
            model_slugs=comp,
        )
        saved.append(plots_dir / f"{tslug}_tsei_three_way_overlay_log.png")
    plot_val_loss_overlay(
        suite,
        title=f"Validation loss — all models ({tlab})",
        save_path=plots_dir / "val_loss_overlay.png",
    )
    saved.append(plots_dir / "val_loss_overlay.png")
    plot_val_loss_overlay_log(
        suite,
        save_path=plots_dir / "val_loss_overlay_log.png",
    )
    saved.append(plots_dir / "val_loss_overlay_log.png")
    plot_training_overlay_train_val(
        suite,
        title=f"Train / val — all models ({tlab})",
        save_path=plots_dir / "train_val_overlay.png",
    )
    saved.append(plots_dir / "train_val_overlay.png")
    plot_training_overlay_train_val_log(
        suite,
        save_path=plots_dir / "train_val_overlay_log.png",
    )
    saved.append(plots_dir / "train_val_overlay_log.png")
    save_per_model_curves(suite, plots_dir)
    for name in suite["histories"]:
        saved.append(plots_dir / f"{name}_loss.png")
        saved.append(plots_dir / f"{name}_loss_log.png")
    if comp:
        plot_model_predictions_grid(
            pack,
            suite,
            y_val,
            input_title="Input (full res.)",
            n_samples=n_pred_samples,
            save_path=plots_dir / f"{tslug}_tsei_prediction_three_way.png",
            suptitle=f"{tlab}: input, target, and selected model predictions",
            target_label=tlab,
            model_slugs=comp,
        )
        saved.append(plots_dir / f"{tslug}_tsei_prediction_three_way.png")
    plot_prediction_row(
        pack,
        suite,
        y_val,
        n_samples=min(3, len(y_val)),
        save_path=plots_dir / "prediction_comparison_all_models.png",
    )
    saved.append(plots_dir / "prediction_comparison_all_models.png")
    b_va = pack.get("boundary_val")
    if (
        b_va is not None
        and len(b_va) > 0
        and pack.get("has_bsds_boundaries")
        and not pack.get("synthetic", True)
    ):
        if comp:
            plot_model_predictions_grid(
                pack,
                suite,
                y_val,
                aux_gt=b_va,
                aux_title="True contours (BSDS)",
                aux_cmap="nipy_spectral",
                input_title="Input (L)",
                n_samples=n_pred_samples,
                save_path=plots_dir / f"{tslug}_prediction_with_bsds_contours.png",
                suptitle=(
                    f"BSDS: input, human boundaries, target ({tlab}), "
                    "TSEI / MyOwn preds"
                ),
                target_label=tlab,
                model_slugs=comp,
            )
            saved.append(plots_dir / f"{tslug}_prediction_with_bsds_contours.png")
        plot_model_predictions_grid(
            pack,
            suite,
            y_val,
            aux_gt=b_va,
            aux_title="True contours (BSDS)",
            aux_cmap="nipy_spectral",
            input_title="Input (L)",
            use_all_models=True,
            n_samples=min(3, len(y_val)),
            save_path=plots_dir / "prediction_all_models_with_bsds_contours.png",
            suptitle=(
                f"Validation: L-channel | human boundaries | {tlab[:40]} | models"
            ),
            target_label=tlab,
        )
        saved.append(plots_dir / "prediction_all_models_with_bsds_contours.png")
    m1 = suite["models"].get("tsei_1block")
    lay = m1.get_layer("tsei_block") if m1 is not None else None
    if m1 is not None and isinstance(lay, TSEIMorphologicalBlock):
        visualize_tsei_pair_grids(
            lay,
            KERNEL_SIZE[0],
            KERNEL_SIZE[1],
            max_filters=6,
            title_prefix="1-block ",
            save_path=plots_dir / "tsei_b_eps_b_delta.png",
        )
        saved.append(plots_dir / "tsei_b_eps_b_delta.png")
    se_paths = plot_suite_structuring_elements(
        suite,
        (KERNEL_SIZE[0], KERNEL_SIZE[1]),
        plots_dir,
        show=show,
        experiment_label=tlab,
    )
    saved.extend(se_paths)
    return [p for p in saved if Path(p).exists()]


def build_experiment_metadata(
    run_dir: Path,
    pack: dict,
    suite: dict,
    *,
    extra: dict | None = None,
) -> dict:
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "synthetic_data": bool(pack.get("synthetic", False)),
        "has_bsds_boundaries": bool(pack.get("has_bsds_boundaries", False)),
        "img_size": [int(pack["H"]), int(pack["W"])],
        "hyperparameters": {
            "epochs": EPOCHS_MAIN,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "kernel_size": list(KERNEL_SIZE),
            "num_filters_tsei": NUM_FILTERS_TSEI,
            "bsds_root": BSDS_ROOT,
            "training_callback_mode": TRAINING_CALLBACK_MODE,
            "val_loss_early_stop_threshold": VAL_LOSS_EARLY_STOP_THRESHOLD,
            "early_stop_monitor": EARLY_STOP_MONITOR,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopping_restore_best_weights": EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
        },
        "checkpoint_root": suite.get("checkpoint_root"),
        "val_mse": dict(suite["val_mse"]),
        "models_trained": list(suite["histories"].keys()),
        "models_to_train": list(
            suite.get("models_to_train", list(suite["histories"].keys()))
        ),
    }
    try:
        import tensorflow as tf

        meta["tensorflow_version"] = tf.__version__
    except Exception:
        pass
    meta["keras_version"] = keras.__version__
    if extra:
        meta.update(extra)
    return meta


def write_metrics_and_metadata(
    run_dir: Path,
    pack: dict,
    suite: dict,
    *,
    target_meta: dict | None = None,
) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    extra = dict(target_meta or {})
    meta = build_experiment_metadata(run_dir, pack, suite, extra=extra)
    metrics = {
        "val_mse": dict(suite["val_mse"]),
        "histories": histories_to_json_safe(suite["histories"]),
    }
    mp = run_dir / "metrics.json"
    with open(mp, "w") as f:
        json.dump(metrics, f, indent=2)
    metap = run_dir / "metadata.json"
    with open(metap, "w") as f:
        json.dump(meta, f, indent=2)
    return mp, metap


def print_experiment_console_summary(
    run_dir: Path,
    pack: dict,
    suite: dict,
    *,
    target_meta: dict | None = None,
    metrics_path: Path | None = None,
    metadata_path: Path | None = None,
) -> None:
    """Print a console résumé aligned with ``metadata.json`` / ``metrics.json`` in ``run_dir``."""
    run_dir = Path(run_dir).resolve()
    tm = dict(target_meta or {})
    tk = tm.get("target_key", "?")
    tlab = tm.get("target_label", TARGET_LABELS.get(str(tk), str(tk)))
    meta = build_experiment_metadata(run_dir, pack, suite, extra=tm)
    hp = meta.get("hyperparameters", {})
    print("\n" + "=" * 72)
    print("Experiment run — summary (same artifacts as output folder)")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    if metadata_path is not None:
        print(f"metadata.json: {Path(metadata_path).resolve()}")
    if metrics_path is not None:
        print(f"metrics.json:  {Path(metrics_path).resolve()}")
    print(f"Target: {tk} — {tlab}")
    print(f"Data: synthetic={meta.get('synthetic_data')}, "
          f"has_bsds_boundaries={meta.get('has_bsds_boundaries')}, "
          f"img_size={meta.get('img_size')}")
    print("Hyperparameters:")
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "kernel_size",
        "num_filters_tsei",
        "training_callback_mode",
        "val_loss_early_stop_threshold",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_monitor",
    ):
        if key in hp:
            print(f"  {key}: {hp[key]}")
    if meta.get("tensorflow_version"):
        print(f"  tensorflow_version: {meta['tensorflow_version']}")
    print(f"  keras_version: {meta.get('keras_version', keras.__version__)}")
    print(f"Models trained: {meta.get('models_to_train', meta.get('models_trained', []))}")
    if meta.get("checkpoint_root"):
        print(f"Checkpoint root: {meta['checkpoint_root']}")
    print(f"plots directory: {run_dir / 'plots'}")
    print("\nValidation MSE (cropped to target grid, best checkpoint if saved):")
    for slug, v in meta.get("val_mse", {}).items():
        disp = MODEL_DISPLAY_NAMES.get(slug, slug)
        print(f"  {disp} ({slug}): {float(v):.6f}")
    print("\nLearnable K constants after training (per model):")
    models = suite.get("models") or {}
    for slug in meta.get("models_trained", []):
        m = models.get(slug)
        if m is None:
            continue
        kmap = extract_k_constants_from_model(m)
        disp = MODEL_DISPLAY_NAMES.get(slug, slug)
        if kmap:
            parts = [f"{name}={val:.6f}" for name, val in kmap.items()]
            print(f"  {disp} ({slug}): " + ", ".join(parts))
        else:
            print(f"  {disp} ({slug}): (none — no global K in this architecture)")
    print(
        "\nmetrics.json stores per-epoch loss / val_loss curves; "
        "checkpoints/ and plots/ are under the run directory above."
    )
    print("=" * 72 + "\n")


def experiment_run_markdown(run_dir: Path, suite: dict, *, target_key: str = "mg9") -> str:
    """Short explanation for notebook display (what each artifact means)."""
    tslug = target_slug_for_files(target_key)
    lines = [
        "### Morphological antidilation experiment run",
        "",
        f"**Run directory:** `{run_dir}`",
        "",
        f"**Target:** `{target_key}` — {TARGET_LABELS.get(target_key, target_key)}",
        "",
        "- **`metadata.json`** — UTC timestamp, data flags, hyperparameters, TensorFlow/Keras versions, and validation MSE per model.",
        "- **`metrics.json`** — Per-epoch `loss` / `val_loss` curves (all models) plus final `val_mse` table.",
        "- **`checkpoints/<model>/best.weights.h5`** — Lowest validation loss during training.",
        "- **`checkpoints/<model>/last.weights.h5`** — Weights after the final epoch (no per-epoch dumps).",
        "- **Early stopping** — Controlled by `TRAINING_CALLBACK_MODE` (`patience` uses Keras `EarlyStopping` with `min_delta` / `patience`; optional fixed threshold). With `restore_best_weights=True`, the fitted model matches the best epoch before `load_best_weights_if_present` reloads from disk.",
        "",
        "**Plots (under `plots/`):**",
        f"- `{tslug}_tsei_three_way_overlay*.png` — Train/val curves for the preferred comparison subset; linear and log.",
        "- `val_loss_overlay*.png` / `train_val_overlay*.png` — All six architectures; linear and log superpositions.",
        "- `*_loss.png` / `*_loss_log.png` — Per-model train/val curves.",
        f"- `{tslug}_tsei_prediction_three_way.png` — Input, supervision target, one column per model in that subset.",
        "- `prediction_comparison_all_models.png` — Same idea including sup-erosion baselines.",
        f"- `{tslug}_prediction_with_bsds_contours.png` / `prediction_all_models_with_bsds_contours.png` — When BSDS `groundTruth` is available: L-channel, **human boundary map**, target, then model preds.",
        "- `tsei_b_eps_b_delta.png` — Learned ε / δ* filters for the 1-block TSEI.",
        "- `tsei*_*.png`, `baseline_*.png`, `single_sup_erosion.png` — Structuring-element grids.",
        "",
        "**Validation MSE (best checkpoint weights):**",
    ]
    for k, v in suite["val_mse"].items():
        lines.append(f"- `{k}`: {v:.6f}")
    return "\n".join(lines)


def maybe_display_run_in_notebook(
    run_dir: Path,
    suite: dict,
    *,
    notebook: bool,
    target_key: str = "mg9",
) -> None:
    if not notebook:
        return
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        return
    display(Markdown(experiment_run_markdown(run_dir, suite, target_key=target_key)))
    plots_dir = run_dir / "plots"
    if not plots_dir.is_dir():
        return
    for p in sorted(plots_dir.glob("*.png")):
        display(Image(filename=str(p)))


def show_plots_from_saved_run(
    run_dir: Path | str,
    *,
    notebook: bool = True,
    plot_subdir: str | None = "replay",
    target_key: str | None = None,
) -> None:
    """Reload best checkpoints + ``metrics.json``, redraw comparisons into ``plots/`` (or ``plots/<plot_subdir>``).

    Default ``plot_subdir=\"replay\"`` avoids overwriting PNGs written during training; pass ``None`` to
    refresh files in ``plots/`` in place.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    base = run_dir / "plots"
    plots_out = base if plot_subdir is None else base / plot_subdir
    pack, y_va, suite, run_meta = load_suite_for_visualization(run_dir, target_key=target_key)
    tk = (target_key or run_meta.get("target_key") or "mg9").lower().strip()
    tlab = run_meta.get("target_label") or TARGET_LABELS.get(tk, tk)
    import matplotlib

    matplotlib.use("Agg")
    run_comparison_plots(
        suite,
        pack,
        y_va,
        plots_out,
        target_key=tk,
        target_label=tlab,
        show=False,
        n_pred_samples=min(4, len(y_va)),
    )
    if not notebook:
        return
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        return
    display(Markdown(experiment_run_markdown(run_dir, suite, target_key=tk)))
    display(Markdown(f"**Figures:** `{plots_out}`"))
    for p in sorted(plots_out.glob("*.png")):
        display(Image(filename=str(p)))


# %% [markdown] id="493a016f"
# ## 10. Interpretation
# - **Antidilation vs supremum-of-erosions:** TSEI and the MyOwn baseline explicitly pair an erosion-like
#   branch with an antidilation branch before the outer supremum; the sup-erosion baselines stack
#   supremum-of-erosions maps without that paired antidilation structure (except as implicit in deeper
#   combining layers).
# - **Depth:** A second TSEI block learns a map on the **feature map** produced by the first block (valid
#   geometry); two-layer models here combine parallel first-stage maps or masked receptive fields.
# - **Scale $K$:** The learnable scalar in `TSEIMorphologicalBlock` (and `ScaledOutput` in the MyOwn
#   baseline) absorbs global amplitude, which helps match gradient magnitudes when patch-wise operations
#   are only approximately 1-Lipschitz in practice.

# %% [markdown] id="3717641a"
# ## 11. Experiment driver
#
# The next code cell defines only **`run_experiment`**. Below that: a **step-by-step explanation** of what it
# does, then **one code cell per supervised target** (`target_key`). **`main`** / **`if __name__`** live at the
# **end** (§11c)—CLI only, not extra benchmarks. Run **only** one target cell at a time. For post-hoc plots,
# use §11b.

# %% colab={"base_uri": "https://localhost:8080/"} id="fc7e2acc" outputId="6d0b4d74-1b70-4c43-a519-c94b6c2a3e2d"
def run_experiment(
    target_key: str = "mg9",
    out_dir: Path | str | None = None,
    *,
    models_to_train: Sequence[str] | None = None,
    notebook: bool = False,
    dated_subdir: bool = True,
) -> dict:
    """Train selected architectures on ``TARGET_LABELS[target_key]``, save checkpoints, plots, and JSON.

    Parameters
    ----------
    target_key
        One of :func:`list_target_keys` — e.g. ``\"mg9\"``, ``\"horizontal_diff_half\"``,
        ``\"four_pixel_maxmin\"``, ``\"internal_gradient_pair\"``, ``\"external_gradient_pair\"``.
    models_to_train
        Subset of :data:`ALL_MODEL_SLUGS`; default :data:`DEFAULT_EXPERIMENT_MODELS` (TSEI 1/2 + MyOwn,
        no supremum baselines). Example — full zoo:
        ``(\"tsei_1block\", \"tsei_2block\", \"myown_tsei\", \"single_sup\", \"two_layer_sup\", \"two_layer_rf\")``.
    """
    import matplotlib

    matplotlib.use("Agg")

    parent = Path(out_dir or "_outputs").expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f"exp_{target_slug_for_files(target_key)}"
    if dated_subdir:
        run_dir = new_dated_experiment_dir(parent, prefix)
    else:
        run_dir = parent
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    plots_dir = run_dir / "plots"
    ck_root = run_dir / "checkpoints"

    pack = resolve_data_pack()
    x_tr = pack["x_train"]
    y_tr, y_va, tmeta = compute_targets_for_experiment(pack, target_key)

    print("Target:", tmeta["target_key"], "—", tmeta["target_label"])
    print("Models:", list(parse_models_to_train(models_to_train)))
    print("Shapes:", x_tr.shape, y_tr.shape, y_va.shape)
    print("Experiment run directory:", run_dir)

    suite = train_and_eval_suite(
        pack,
        y_tr,
        y_va,
        epochs=EPOCHS_MAIN,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        verbose=1,
        checkpoint_root=ck_root,
        models_to_train=models_to_train,
    )
    suite["experiment_run_dir"] = str(run_dir.resolve())
    mp, metap = write_metrics_and_metadata(run_dir, pack, suite, target_meta=tmeta)
    run_comparison_plots(
        suite,
        pack,
        y_va,
        plots_dir,
        target_key=tmeta["target_key"],
        target_label=tmeta["target_label"],
        show=False,
        n_pred_samples=min(4, len(y_va)),
    )
    print_experiment_console_summary(
        run_dir,
        pack,
        suite,
        target_meta=tmeta,
        metrics_path=mp,
        metadata_path=metap,
    )
    maybe_display_run_in_notebook(
        run_dir, suite, notebook=notebook, target_key=tmeta["target_key"]
    )
    return suite


# %% [markdown] id="exp-md-how-run-experiment-works"
# ### 11a. How `run_experiment` works
#
# **`run_experiment(target_key, ...)`** runs the full training + logging pipeline for one choice of
# **supervised target** (see §5 and `TARGET_LABELS`). In order:
#
# 1. **`matplotlib.use("Agg")`** — non-interactive backend; figures are written as PNGs (the notebook can
#    display them via `maybe_display_run_in_notebook` when `notebook=True`).
# 2. **Run directory** — `out_dir` (default `_outputs`) plus a timestamped subfolder
#    `exp_<target_slug>_YYYY-MM-DD_HHMMSS` when `dated_subdir=True`, containing `plots/` and `checkpoints/`.
# 3. **Data** — `resolve_data_pack()` loads BSDS or synthetic images (`x_train`, `x_val`, …).
# 4. **Targets** — `compute_targets_for_experiment(pack, target_key)` builds `y_train`, `y_val` and crops
#    them to the **one-block TSEI** valid grid; metadata (`target_key`, shapes) is merged into JSON later.
# 5. **Training** — `train_and_eval_suite(..., models_to_train=...)` fits only the architectures you list
#    (default `DEFAULT_EXPERIMENT_MODELS`: TSEI 1-block, TSEI 2-block, MyOwn). Checkpoints:
#    `checkpoints/<model>/best.weights.h5` and `last.weights.h5`.
# 6. **Artifacts** — `write_metrics_and_metadata` writes `metadata.json` (hyperparameters, `models_to_train`,
#    versions) and `metrics.json` (per-epoch `loss` / `val_loss` per model, final `val_mse`).
# 7. **Figures** — `run_comparison_plots` saves overlays, per-model curves, and prediction comparisons under
#    `plots/`.
# 8. **Inline display** — if `notebook=True`, `maybe_display_run_in_notebook` shows a short Markdown summary
#    and each PNG in the notebook.
#
# **Returns** the `suite` dict (`histories`, `models`, `val_mse`, …) and sets `suite["experiment_run_dir"]`.
# Optional: pass `models_to_train=list(ALL_MODEL_SLUGS)` to include supremum-of-erosions baselines.
#
# **`main()`** (defined at the **end** of this file, §11c) is only for `python script.py` / Colab batch mode:
# it calls `run_experiment("mg9", ...)` once with `notebook=False`. In Jupyter, use the per-target cells
# below instead—do not confuse §11c with those benchmarks.
#
# **Tip:** Execute **one** target cell at a time below; each run trains all selected models and can take a
# long time. Do not “Run All” unless you intend to queue every benchmark.

# %% [markdown] id="exp-md-target-mg9"
# #### Experiment — `mg9` (vertical $|u{-}v|$ on the $3\times 3$ stencil)

# %% id="exp-code-target-mg9"
suite_mg9 = run_experiment(
    "mg9",
    out_dir="_outputs",
    notebook=True,
    dated_subdir=True,
)
RUN_DIR_MG9 = Path(suite_mg9["experiment_run_dir"])

# %% [markdown] id="exp-md-target-horizontal-diff"
# #### Experiment — `horizontal_diff_half`: ($m(x_1,x_2)=\tfrac{1}{2}(x_1-x_2)$)

# %% id="exp-code-target-horizontal-diff"
suite_horizontal_diff = run_experiment(
    "horizontal_diff_half",
    out_dir="_outputs",
    notebook=True,
    dated_subdir=True,
)
RUN_DIR_HORIZONTAL_DIFF = Path(suite_horizontal_diff["experiment_run_dir"])

# %% [markdown] id="exp-md-target-four-pixel"
# #### Experiment — `four_pixel_maxmin` (max − min on each $2\times 2$ patch)

# %% id="exp-code-target-four-pixel"
suite_four_pixel_maxmin = run_experiment(
    "four_pixel_maxmin",
    out_dir="_outputs",
    notebook=True,
    dated_subdir=True,
)
RUN_DIR_FOUR_PIXEL = Path(suite_four_pixel_maxmin["experiment_run_dir"])

# %% [markdown] id="exp-md-target-internal-pair"
# #### Experiment — `internal_gradient_pair`: \(x_1 - \min(x_1,x_2)\) (horizontal neighbors)

# %% id="exp-code-target-internal-pair"
suite_internal_pair = run_experiment(
    "internal_gradient_pair",
    out_dir="_outputs",
    notebook=True,
    dated_subdir=True,
)
RUN_DIR_INTERNAL_PAIR = Path(suite_internal_pair["experiment_run_dir"])

# %% [markdown] id="exp-md-target-external-pair"
# #### Experiment — `external_gradient_pair`: \(\max(x_1,x_2) - x_1\) (horizontal neighbors)

# %% id="exp-code-target-external-pair"
suite_external_pair = run_experiment(
    "external_gradient_pair",
    out_dir="_outputs",
    notebook=True,
    dated_subdir=True,
)
RUN_DIR_EXTERNAL_PAIR = Path(suite_external_pair["experiment_run_dir"])

# %% [markdown] id="exp-md-post-training"
# ### 11b. Post-training: reload best checkpoints and redraw plots
#
# Set `RUN_DIR` to a folder produced by `run_experiment` (it must contain `metadata.json`, `metrics.json`,
# and `checkpoints/*/best.weights.h5`). **`show_plots_from_saved_run`** rebuilds models, loads **best**
# weights, reads curves from `metrics.json`, and redraws comparisons. Default output: `plots/replay/` so
# training-time PNGs are not overwritten.

# %%
# from pathlib import Path
# RUN_DIR = Path("_outputs") / "exp_mg9_2026-01-01_000000"  # set to your latest run folder

# %%
# show_plots_from_saved_run(RUN_DIR, notebook=True, plot_subdir="replay")

# %% [markdown] id="exp-md-cli-entry"
# ### 11c. Command-line entry (`main`) — not a notebook benchmark
#
# This block exists so you can run **`python exp_antidilation_morpho_gradients.py`** (or Colab) and execute a
# single default training run: **`run_experiment("mg9", ...)`** with `notebook=False`. It is **not** one of the
# supervised-target experiments above; those are driven by the **`run_experiment(target_key=...)`** cells in
# §11a. Keep this section **last** so all notebook experiments stay grouped above.

# %% id="exp-code-cli-main"
def main(
    out_dir: Path | str | None = None,
    *,
    notebook: bool = False,
    dated_subdir: bool = True,
) -> dict:
    """CLI / script entry: one `mg9` run (same as ``run_experiment('mg9', ...)``)."""
    return run_experiment(
        "mg9", out_dir=out_dir, notebook=notebook, dated_subdir=dated_subdir
    )


if __name__ == "__main__":
    main(out_dir="/content/outputs", notebook=False, dated_subdir=True)

# %% id="5hN9eK0F_O46"
# !cp -r '/content/outputs/' '/content/drive/MyDrive/experiments-dima/outputs/'
