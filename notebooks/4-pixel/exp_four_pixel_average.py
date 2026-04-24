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

# %%
from __future__ import annotations

# %% [markdown]
# # Four-pixel average — supremum-of-erosions experiments
#
# Learn the **4-pixel average** filter with trainable **sup**-erosion networks; ground truth comes from a fixed
# conv layer.

# %% [markdown]
# ## 1. Mathematical background
#
# We supervise toward the **4-pixel average** (cross-shaped $3\times3$ stencil): same-padding convolution with
# $0.25$ on the four orthogonal neighbors and zero at the center.
#
# **Dilation** means, for each filter, the max over the patch of $(f + w)$ with learnable offsets $w$.
# **Erosion** is the standard dual, $-(({-f}) \oplus w)$, then we take **sup** over filters.
#
# Two-layer layouts use two parallel erosion banks and a **combiner** (per-filter minimum of the two shifted
# maps, then maximum over filters). **Receptive-field** variants mask some patch weights to a large negative
# constant so those stencil entries are effectively inactive.

# %% [markdown]
# ### 1.5 Experimental protocol — From the paper *Optimizing Morphological Representations: Robustness to Initialization and Gradient Sparsity*
#
# The benchmark studies optimization in morphological layers: how **initialization** and **gradient sparsity**
# affect learning a fixed image transformation. This notebook instantiates the **4-pixel average** target; the
# layout matches the paper's noisy **Fashion-MNIST** setup and MSE training recipe.
#
# **Data and preprocessing**
#
# - **Dataset:** Noisy **Fashion-MNIST** (small subsets; see **3. Dataahhhset**).
# - **Noise:** Zero-mean Gaussian with $\sigma = 40$ on **8-bit** pixel values, then **clip** to $[0, 255]$.
# - **Normalization:** Divide by **255** so model inputs live in **$[0, 1]$** (NHWC).
# - **Split:** First **100** training images and first **100** test images (test slice used as validation); noise
#   is drawn with a **fixed RNG seed** so protocol repetitions share the same noisy data while **weight
#   initialization** varies (see **`DATA_LOAD_SEED`** vs per-rep seeds in **`run_experiment_repeated`**).
#
# **Architecture and fixed hyperparameters (this notebook)**
#
# - **Sup-erosions:** Implemented in Keras (**`MyOwnDilation`** + **`SupErosions_block`** where needed).
# - **Number of structuring elements ($K$):** **1200** filters in the single-layer model (**`N_EROSIONS_SINGLE`**);
#   two-layer variants use **`N_EROSIONS_TL1`**, **`N_EROSIONS_TL2`**, **`N_EROSIONS_TL3`**.
# - **Stencil size:** **$3 \times 3$** (**`KERNEL_SIZE`**).
# - **Batch size:** **50** (**`BATCH_SIZE`**).
# - **Loss:** **MSE** (in-graph **`mse_crop_pred_to_true`**; targets in $[0,1]$).
# - **Optimizer:** **Adam**, **`LEARNING_RATE`** $= 0.01$.
# - **Training length:** Up to **2000** epochs (**`EPOCHS_MAIN`**), with early stopping when **val_loss** falls
#   below $Q = 1/255^2$ (**`VAL_LOSS_EARLY_STOP_THRESHOLD`**, one gray-level scale on normalized inputs).
# - **Training rule:** This notebook currently uses only the standard optimization step.
#
# **Varying initialization (paper vs this file)**
#
# - Weights use **uniform** random values in **`[INIT_MIN, INIT_MAX]`**, derived from **`INIT_CENTER`** and
#   **`INIT_HALF_WIDTH`** so you can sweep the interval center without editing layer code.
# - The paper repeats training **20** times per initialization setting; here **`N_TRAINING_REPETITIONS`** defaults
#   to **3** for speed—raise it when you need means and standard deviations closer to the full protocol.
#
# **Evaluation (aligned with the paper)**
#
# 1. **Success rate:** Fraction of repetitions where final **validation MSE** is below **$Q$**.
# 2. **Minimal structuring elements:** Count after **Pareto** reduction of learned filters (analysis and plots).


# %% [markdown]
# ## 2. Environment and setup
#
# This section pulls in dependencies and fixes the numeric knobs used everywhere else: image size, kernel shape,
# dataset size and noise, counts of learnable erosion filters, optimization schedule, initializer range, protocol
# **repetitions**, data vs training RNG seeds, early stopping, and receptive-field mask indices. The global
# **Keras** seed is set from **`RANDOM_SEED`** for a single full run; **`run_experiment_repeated`** resets it per
# repetition so only initialization changes, not the noisy dataset.

# %%
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

# %%
RANDOM_SEED = 42
keras.utils.set_random_seed(RANDOM_SEED)

DATA_LOAD_SEED = RANDOM_SEED
N_TRAINING_REPETITIONS = 3
TRAINING_SEED_STEP = 10_000

IMG_H, IMG_W = 28, 28
KERNEL_SIZE: tuple[int, int] = (3, 3)

DATASET_SIZE = 100
NOISE_SIGMA = 40.0
USE_NOISE = True

N_EROSIONS_SINGLE = 1200
N_EROSIONS_TL1 = 500
N_EROSIONS_TL2 = 500
N_EROSIONS_TL3 = 700

EPOCHS_MAIN = 2000
BATCH_SIZE = 50
LEARNING_RATE = 0.01

INIT_CENTER = 0.0
INIT_HALF_WIDTH = 0.35
INIT_MIN = INIT_CENTER - INIT_HALF_WIDTH
INIT_MAX = INIT_CENTER + INIT_HALF_WIDTH

QFACT = 255.0
VAL_LOSS_EARLY_STOP_THRESHOLD = 1.0 / QFACT**2

TRAINING_CALLBACK_MODE: str = "both"
EARLY_STOPPING_PATIENCE: int = 25
EARLY_STOPPING_MIN_DELTA: float = 1e-5
EARLY_STOPPING_RESTORE_BEST_WEIGHTS: bool = True

# ``standard`` — plain backprop.
TRAINING_UPDATE_RULE: str = "standard"
SPARSE_SUBMODEL_MASK_MIN: float = 10.0
SPARSE_SUBMODEL_MASK_MAX: float = 10.0
SPARSE_GRAD_ZERO_ATOL: float = 1e-12

RF_BLOCK1_INACTIVE = [4, 5, 6, 7, 8]
RF_BLOCK2_INACTIVE = [0, 1, 2, 3]
RF_INACTIVE_VALUE = -10.0

# %% [markdown]
# ## 3. Dataset
#
# We load **Fashion-MNIST**, take fixed-size train / validation / test slices, optionally add Gaussian noise on
# the original $[0,255]$ pixel scale, then clip and convert to **NHWC** tensors in $[0,1]$. The pack also keeps
# clean copies when noise is on, which is useful for side-by-side visualization.

# %%
def _add_gaussian_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = x.astype(np.float32) + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.clip(out, 0.0, 255.0)


def load_fashion_mnist_four_pixel_pack(
    dataset_size: int = DATASET_SIZE,
    noise_sigma: float = NOISE_SIGMA,
    use_noise: bool = USE_NOISE,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_clean_train = x_train[:dataset_size]
    x_clean_val = x_test[:dataset_size]
    x_clean_test = x_test[dataset_size : 2 * dataset_size]

    if use_noise:
        x_train_n = _add_gaussian_noise(x_clean_train, noise_sigma, rng)
        x_val_n = _add_gaussian_noise(x_clean_val, noise_sigma, rng)
        x_test_n = _add_gaussian_noise(x_clean_test, noise_sigma, rng)
    else:
        x_train_n = x_clean_train
        x_val_n = x_clean_val
        x_test_n = x_clean_test

    def to_nhwc(x: np.ndarray) -> np.ndarray:
        return x[..., np.newaxis] / 255.0

    return {
        "x_train": to_nhwc(x_train_n),
        "x_val": to_nhwc(x_val_n),
        "x_test": to_nhwc(x_test_n),
        "x_train_clean": to_nhwc(x_clean_train),
        "H": IMG_H,
        "W": IMG_W,
        "synthetic": False,
        "dataset": "fashion_mnist_noisy" if use_noise else "fashion_mnist",
    }


# %% [markdown]
# ## 4. Targets
#
# Ground truth is one `Conv2D` with fixed **4-pixel average** weights, then `predict` on normalized inputs.
# Below: **conv / model_gt**, stacked **Y** tensors, and helpers that bundle labels for training and metadata.

# %% [markdown] id="5KEFBcgrdlCK"
# The theoretical representation of this filter resembles to:
#
# [0, $r_1$, 0]
# [$r_2$, 0, $r_3$]
# [0, -$r_1$-$r_2$-$r_3$, 0]    
# with $r_1,r_2,r_3 \in R$.


# %% [markdown]
# ### 4.1 Ground truth — `model_gt` and 4-pixel average labels
#
# The stencil weights live in **`convolution_matrix`**; **`build_gt_conv_model`** wires them into a single
# `Conv2D` with **same** padding and no bias. **`compute_targets_four_pixel_average`** runs **`predict`** on the
# normalized inputs to produce **`y_train`**, **`y_val`**, and **`y_test`**.

# %%
# In order to create the dataset for this task, we will use a model with a single convolutional layer and parameters :

convolution_matrix = np.array(
    [[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]],
    dtype=np.float32,
)

# cross_filter /= cross_filter.sum()  # Normalization

convolution_matrix = convolution_matrix.reshape((3, 3, 1, 1))


def build_gt_conv_model() -> keras.Model:
    # Define a simple model with one convolutional layer model_gt — the model used to generate the ground truth
    model_gt = keras.Sequential(
        [
            keras.layers.Input(shape=(IMG_H, IMG_W, 1)),
            keras.layers.Conv2D(
                filters=1, kernel_size=(3, 3), padding="same", use_bias=False
            ),
        ]
    )
    model_gt.layers[0].set_weights([convolution_matrix])
    return model_gt


def compute_targets_four_pixel_average(pack: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_gt = build_gt_conv_model()
    # Creating the Ground Truth Datasets (inputs already in [0,1])
    y_tr = model_gt.predict(pack["x_train"], verbose=0)
    y_va = model_gt.predict(pack["x_val"], verbose=0)
    y_te = model_gt.predict(pack["x_test"], verbose=0)
    return y_tr.astype(np.float32), y_va.astype(np.float32), y_te.astype(np.float32)


# %% [markdown]
# ### 4.2 Labels and experiment metadata
#
# **`EXPERIMENT_TARGET_SLUG`** names run directories; **`EXPERIMENT_TARGET_LABEL`** is the caption text for plots
# and **`metadata.json`**. **`compute_experiment_targets`** calls **`compute_targets_four_pixel_average`** and
# returns a small **meta** dict with tensor shapes plus those two strings.

# %%
EXPERIMENT_TARGET_SLUG = "four_pixel_average"

EXPERIMENT_TARGET_LABEL = "4-pixel average output (cross conv, same padding)"


def compute_experiment_targets(
    pack: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    y_tr, y_va, y_te = compute_targets_four_pixel_average(pack)
    meta = {
        "target_key": EXPERIMENT_TARGET_SLUG,
        "target_label": EXPERIMENT_TARGET_LABEL,
        "y_train_shape": tuple(int(x) for x in y_tr.shape),
        "y_val_shape": tuple(int(x) for x in y_va.shape),
    }
    return y_tr, y_va, y_te, meta


# %% [markdown]
# ## 5. Models
#
# **`MyOwnDilation`** (patch max + learnable offsets), a **combiner** block for two-branch layouts, then **three**
# experiment architectures (single sup-erosions, two-layer, two-layer with receptive-field masks).

# %% [markdown]
# ### 5.1 Implementing dilation layer with keras — `MyOwnDilation`
#
# For each spatial location and filter, the layer extracts a patch, adds a trainable offset vector **`w`**, and
# takes the **maximum** over patch entries. That is the discrete max-plus analogue of grayscale dilation.

# %%
class MyOwnDilation(keras.layers.Layer):
    def __init__(
        self,
        filters=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="VALID",
        minval=-0.35,
        maxval=0.35,
        seed=None,
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

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(
                1,
                1,
                1,
                self.kernel_size[0] * self.kernel_size[1],
                self.filters,
            ),
            initializer=keras.initializers.RandomUniform(
                minval=self.minval, maxval=self.maxval, seed=self.seed
            ),
            trainable=True,
        )

    def call(self, inputs):
        N = keras.ops.image.extract_patches(
            inputs,
            self.kernel_size,
            strides=self.stride,
            dilation_rate=1,
            padding=self.padding,
        )
        N = keras.ops.expand_dims(N, axis=-1)
        return keras.ops.max(N + self.w, axis=3)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "minval": self.minval,
                "maxval": self.maxval,
                "seed": self.seed,
            }
        )
        return config


# %% [markdown]
# ### 5.2 Combiner block — `SupErosions_block` (two maps, then sup over filters)
#
# The block accepts two single-channel feature maps, subtracts one scalar weight per filter and branch, applies a
# **per-filter minimum** (erosion-like), then a **maximum over filters** (supremum of those erosions).

# %%
class SupErosions_block(keras.layers.Layer):
    def __init__(self, n_erosions, minval=-0.35, maxval=0.35, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.n_erosions = n_erosions
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def build(self, input_shape):
        self.w1 = self.add_weight(
            shape=(self.n_erosions,),
            initializer=keras.initializers.RandomUniform(
                minval=self.minval, maxval=self.maxval, seed=self.seed
            ),
            trainable=True,
        )
        self.w2 = self.add_weight(
            shape=(self.n_erosions,),
            initializer=keras.initializers.RandomUniform(
                minval=self.minval, maxval=self.maxval, seed=self.seed
            ),
            trainable=True,
        )

    def call(self, inputs):
        input1, input2 = inputs
        w1b = keras.ops.reshape(self.w1, (1, 1, 1, -1))
        w2b = keras.ops.reshape(self.w2, (1, 1, 1, -1))
        z1 = input1 - w1b
        z2 = input2 - w2b
        t = keras.ops.minimum(z1, z2)  # erosions
        return keras.ops.max(t, axis=-1, keepdims=True)  # supremum of the erosions

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_erosions": self.n_erosions,
                "minval": self.minval,
                "maxval": self.maxval,
                "seed": self.seed,
            }
        )
        return config


def apply_k_deactivated_structuring_elements_init(
    model: keras.Model,
    *,
    k_deactivated_components: int = 0,
    inactive_value: float = -10.0,
) -> None:
    """
    Set `k` random patch components to `inactive_value` in each structuring
    element (filter) of every `MyOwnDilation` layer in `model`.
    """
    k = int(k_deactivated_components)
    if k <= 0:
        return
    for layer in model.layers:
        if not isinstance(layer, MyOwnDilation):
            continue
        w = np.asarray(layer.get_weights()[0], dtype=np.float32)
        patch_size = int(w.shape[3])
        if k > patch_size:
            raise ValueError(
                f"k_deactivated_components={k} is larger than patch size {patch_size}."
            )
        n_filters = int(w.shape[4])
        for filt in range(n_filters):
            inactive_idx = np.random.choice(patch_size, size=k, replace=False)
            w[0, 0, 0, inactive_idx, filt] = inactive_value
        layer.set_weights([w])


# %% [markdown]
# ### 5.3 One-layer SupErosions model
#
# One bank of erosions on the padded input, then **sup** over all filters into a single output channel.

# %%
class SingleSupErosionsArchitecture:
    slug = "single_sup"
    display_name = "Single-layer supremum of erosions"

    @classmethod
    def build(
        cls,
        input_shape: tuple[int, int, int],
        kernel_size: tuple[int, int],
        *,
        k_deactivated_components: int = 0,
    ) -> keras.Model:
        # The SupErosions model (symmetric pad so valid 3x3 covers the full image)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        input_im = keras.layers.Input(shape=input_shape, name="inputLayer")
        input_padding = keras.layers.ZeroPadding2D(padding=padding)(input_im)
        out_erosions = -MyOwnDilation(
            filters=N_EROSIONS_SINGLE,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding="VALID",
            minval=INIT_MIN,
            maxval=INIT_MAX,
            seed=RANDOM_SEED,
            name="Erosion",
        )(-input_padding)
        xout = keras.ops.max(out_erosions, axis=-1, keepdims=True)
        model = keras.Model(input_im, xout, name=cls.slug)
        apply_k_deactivated_structuring_elements_init(
            model, k_deactivated_components=k_deactivated_components
        )
        return model


# %% [markdown]
# ### 5.4 Two parallel SupErosions banks + combiner
#
# Two independent erosion banks read the same padded map; each bank is reduced with **sup**, then
# **`SupErosions_block`** merges the two scalar fields.

# %%
class TwoLayerSupErosionsArchitecture:
    slug = "two_layer_sup"
    display_name = "Two-layer sup-erosions (parallel banks + combiner)"

    @classmethod
    def build(
        cls,
        input_shape: tuple[int, int, int],
        kernel_size: tuple[int, int],
        *,
        k_deactivated_components: int = 0,
    ) -> keras.Model:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        input_im = keras.layers.Input(shape=input_shape, name="inputLayer")
        input_padding = keras.layers.ZeroPadding2D(padding=padding)(input_im)
        out_erosions1 = -MyOwnDilation(
            filters=N_EROSIONS_TL1,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding="VALID",
            minval=INIT_MIN,
            maxval=INIT_MAX,
            seed=RANDOM_SEED,
            name="Erosions1",
        )(-input_padding)
        sup_erosions1 = keras.ops.max(out_erosions1, axis=-1, keepdims=True)
        out_erosions2 = -MyOwnDilation(
            filters=N_EROSIONS_TL2,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding="VALID",
            minval=INIT_MIN,
            maxval=INIT_MAX,
            seed=RANDOM_SEED,
            name="Erosions2",
        )(-input_padding)
        sup_erosions2 = keras.ops.max(out_erosions2, axis=-1, keepdims=True)
        xout = SupErosions_block(
            n_erosions=N_EROSIONS_TL3,
            minval=INIT_MIN,
            maxval=INIT_MAX,
            seed=RANDOM_SEED,
            name="SupErosions_3",
        )([sup_erosions1, sup_erosions2])
        model = keras.Model(input_im, xout, name=cls.slug)
        apply_k_deactivated_structuring_elements_init(
            model, k_deactivated_components=k_deactivated_components
        )
        return model


# %% [markdown]
# ### 5.5 Two-layer model with receptive-field specialization
#
# Same topology as the two-bank model in **5.4**, but after **`build`** we overwrite selected entries of the dilation weights so certain
# patch positions in each bank stay inactive (large negative), nudging the two banks toward different stencils.

# %%
def _apply_receptive_field_masks(
    model: keras.Model,
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


class TwoLayerReceptiveFieldArchitecture:
    slug = "two_layer_rf"
    display_name = "Two-layer sup-erosions + receptive-field masks"

    @classmethod
    def build(
        cls,
        input_shape: tuple[int, int, int],
        kernel_size: tuple[int, int],
        *,
        k_deactivated_components: int = 0,
    ) -> keras.Model:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        input_im = keras.layers.Input(shape=input_shape, name="inputLayer")
        input_padding = keras.layers.ZeroPadding2D(padding=padding)(input_im)
        dil1 = MyOwnDilation(
            filters=N_EROSIONS_TL1,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding="VALID",
            minval=INIT_MIN,
            maxval=INIT_MAX,
            seed=RANDOM_SEED,
            name="Erosions1",
        )
        out_erosions1 = -dil1(-input_padding)
        sup_erosions1 = keras.ops.max(out_erosions1, axis=-1, keepdims=True)
        dil2 = MyOwnDilation(
            filters=N_EROSIONS_TL2,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding="VALID",
            minval=INIT_MIN,
            maxval=INIT_MAX,
            seed=RANDOM_SEED,
            name="Erosions2",
        )
        out_erosions2 = -dil2(-input_padding)
        sup_erosions2 = keras.ops.max(out_erosions2, axis=-1, keepdims=True)
        xout = SupErosions_block(
            n_erosions=N_EROSIONS_TL3,
            minval=INIT_MIN,
            maxval=INIT_MAX,
            seed=RANDOM_SEED,
            name="SupErosions_3",
        )([sup_erosions1, sup_erosions2])
        model = keras.Model(input_im, xout, name=cls.slug)
        model.build((None,) + input_shape)
        apply_k_deactivated_structuring_elements_init(
            model, k_deactivated_components=k_deactivated_components
        )
        _apply_receptive_field_masks(
            model,
            RF_BLOCK1_INACTIVE,
            RF_BLOCK2_INACTIVE,
            RF_INACTIVE_VALUE,
        )
        return model


# %% [markdown]
# ### 5.6 Default run order and helpers
#
# **`ALL_ARCHITECTURE_CLASSES`** fixes the canonical training order; **`ARCHITECTURE_BY_SLUG`** and
# **`parse_models_to_train`** let you restrict or reorder runs by slug while keeping validation strict.
# **`comparison_plot_slugs`** picks which trained models appear in combined figures.

# %%
ALL_ARCHITECTURE_CLASSES: tuple[type, ...] = (
    SingleSupErosionsArchitecture,
    TwoLayerSupErosionsArchitecture,
    TwoLayerReceptiveFieldArchitecture,
)

ARCHITECTURE_BY_SLUG: dict[str, type] = {c.slug: c for c in ALL_ARCHITECTURE_CLASSES}

PREFERRED_MODEL_ORDER: tuple[str, ...] = tuple(c.slug for c in ALL_ARCHITECTURE_CLASSES)


def parse_models_to_train(models: Sequence[str] | None) -> list[type]:
    if models is None:
        return list(ALL_ARCHITECTURE_CLASSES)
    seen: set[str] = set()
    out: list[type] = []
    for s in models:
        key = str(s).strip()
        if key not in ARCHITECTURE_BY_SLUG:
            raise ValueError(
                f"Unknown model slug {key!r}. Use one of: {list(ARCHITECTURE_BY_SLUG.keys())}"
            )
        if key not in seen:
            seen.add(key)
            out.append(ARCHITECTURE_BY_SLUG[key])
    return out


def comparison_plot_slugs(suite: dict) -> list[str]:
    have = suite.get("models") or {}
    return [s for s in PREFERRED_MODEL_ORDER if s in have]


def canonical_architecture_slug(model_key: str) -> str:
    """Map a suite model key (e.g. ``single_sup__kd2``) to a registered architecture slug."""
    if model_key in ARCHITECTURE_BY_SLUG:
        return model_key
    for base in ARCHITECTURE_BY_SLUG:
        if model_key.startswith(f"{base}__kd"):
            return base
    return model_key


def k_deactivated_variant_slug(base_slug: str, k: int) -> str:
    return f"{base_slug}__kd{int(k)}"


# %% [markdown]
# ## 6. Training
#
# This section is split into cells so you can extend **losses**, **optimizers**, **callbacks**, and **training
# loops** without scrolling through one huge block. Flow: metrics → custom callbacks → callback builders → run
# paths / JSON → **`train_model`** → **`train_and_eval_suite`**.

# %% [markdown]
# ### 6.1 Metrics and prediction cropping
#
# **`mse_crop_pred_to_true`** is the compiled loss (graph mode); **`predict_mse`** / **`mse_np`** are for
# post-training evaluation in NumPy.

# %%
def crop_pred_to_y(pred: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
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
    ht = keras.ops.shape(y_true)[1]
    wt = keras.ops.shape(y_true)[2]
    y_pred_c = y_pred[:, :ht, :wt, :]
    return keras.ops.mean(keras.ops.square(y_true - y_pred_c))


def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
    """Pareto-efficient (minimal) rows: no other row dominates coordinate-wise."""
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0
    costs = costs.copy()
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = int(np.sum(nondominated_point_mask[:next_point_index]) + 1)
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    return is_efficient


def iter_my_own_dilation_layers(model: keras.Model) -> list[MyOwnDilation]:
    """``MyOwnDilation`` banks in layer order (PU/DU apply per bank; combiner untouched)."""
    return [layer for layer in model.layers if isinstance(layer, MyOwnDilation)]


STRUCTURING_ELEMENT_PAIR_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "top_corners",
        "title": "(W[0,0], W[0,2])",
        "xlabel": "W[0,0]",
        "ylabel": "W[0,2]",
        "expected_point": (0.0, 0.0),
    },
    {
        "key": "bottom_corners",
        "title": "(W[2,0], W[2,2])",
        "xlabel": "W[2,0]",
        "ylabel": "W[2,2]",
        "expected_point": (0.0, 0.0),
    },
    {
        "key": "center_vs_cross_sum",
        "title": "(W[1,1], W[0,1] + W[1,0] + W[1,2] + W[2,1])",
        "xlabel": "W[1,1]",
        "ylabel": "W[0,1] + W[1,0] + W[1,2] + W[2,1]",
        "expected_point": (0.0, 0.0),
    },
    {
        "key": "cross_sum_vs_max_corners_center",
        "title": "(W[0,1] + W[1,0] + W[1,2] + W[2,1], max{W[0,0],W[0,2],W[1,1],W[2,0],W[2,2]})",
        "xlabel": "W[0,1] + W[1,0] + W[1,2] + W[2,1]",
        "ylabel": "max{W[0,0], W[0,2], W[1,1], W[2,0], W[2,2]}",
        "expected_point": (0.0, -2.0),
        "xlim": (-1.0, 1.0),
        "ylim": (-2.5, 1.0),
    },
)
INDEPENDENT_STRUCTURING_ELEMENT_PAIR_KEY = "cross_sum_vs_max_corners_center"

WEIGHT_PAIR_SNAPSHOT_EPOCHS: tuple[int, ...] = (0, 10, 50, 100, 500)
MINIMAL_ELEMENTS_LOG_EPOCH_STRIDE = 10


def extract_structuring_element_pair_values(weights_5d: np.ndarray) -> dict[str, np.ndarray]:
    """Project each 3x3 structuring element to the requested 2D pair views."""
    flat = np.asarray(weights_5d[0, 0, 0, :, :], dtype=np.float32)
    return {
        "top_corners": np.column_stack([flat[0], flat[2]]).astype(np.float32),
        "bottom_corners": np.column_stack([flat[6], flat[8]]).astype(np.float32),
        "center_vs_cross_sum": np.column_stack(
            [flat[4], flat[1] + flat[3] + flat[5] + flat[7]]
        ).astype(np.float32),
        "cross_sum_vs_max_corners_center": np.column_stack(
            [
                flat[1] + flat[3] + flat[5] + flat[7],
                np.max(
                    np.stack([flat[0], flat[2], flat[4], flat[6], flat[8]], axis=0),
                    axis=0,
                ),
            ]
        ).astype(np.float32),
    }


# %% [markdown]
# ### 6.2 Custom callbacks
#
# Add more **`keras.callbacks.Callback`** subclasses here (logging, LR schedules, custom early stops, etc.).

# %%
class EarlyStopThresholdCallback(keras.callbacks.Callback):
    def __init__(self, loss_threshold: float, monitor: str = "val_loss") -> None:
        super().__init__()
        self.loss_threshold = float(loss_threshold)
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None) -> None:
        del epoch
        logs = logs or {}
        if logs.get(self.monitor, float("inf")) < self.loss_threshold:
            print("\nReached the training threshold")
            self.model.stop_training = True


class SaveLastWeightsCallback(keras.callbacks.Callback):
    def __init__(self, filepath: str | Path) -> None:
        super().__init__()
        self.filepath = Path(filepath)

    def on_train_end(self, logs=None) -> None:
        del logs
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.filepath))


class StructuringElementPairLoggerCallback(keras.callbacks.Callback):
    """Store sparse epoch snapshots of the requested 2D weight projections."""

    def __init__(self) -> None:
        super().__init__()
        self.epochs: list[int] = []
        self._layer_logs: dict[str, dict[str, list[np.ndarray]]] = {}
        self._layer_weight_logs: dict[str, list[np.ndarray]] = {}
        self._snapshot_epochs = set(WEIGHT_PAIR_SNAPSHOT_EPOCHS)
        self._last_epoch_seen = 0

    def on_train_begin(self, logs=None) -> None:
        del logs
        self.epochs = []
        self._layer_logs = {
            layer.name: {spec["key"]: [] for spec in STRUCTURING_ELEMENT_PAIR_SPECS}
            for layer in iter_my_own_dilation_layers(self.model)
        }
        self._layer_weight_logs = {
            layer.name: [] for layer in iter_my_own_dilation_layers(self.model)
        }
        self._last_epoch_seen = 0
        self._snapshot(epoch=0)

    def on_epoch_end(self, epoch, logs=None) -> None:
        del logs
        epoch_num = int(epoch) + 1
        self._last_epoch_seen = epoch_num
        if epoch_num in self._snapshot_epochs:
            self._snapshot(epoch=epoch_num)

    def on_train_end(self, logs=None) -> None:
        del logs
        if self._last_epoch_seen not in self.epochs:
            self._snapshot(epoch=self._last_epoch_seen)

    def _snapshot(self, epoch: int) -> None:
        self.epochs.append(int(epoch))
        for layer in iter_my_own_dilation_layers(self.model):
            weights = np.asarray(layer.get_weights()[0], dtype=np.float32)
            pair_values = extract_structuring_element_pair_values(weights)
            for key, values in pair_values.items():
                self._layer_logs[layer.name][key].append(np.array(values, copy=True))
            self._layer_weight_logs[layer.name].append(np.array(weights, copy=True))

    def export_data(self) -> dict[str, Any]:
        return {
            "epochs": np.asarray(self.epochs, dtype=np.int32),
            "layers": {
                layer_name: {
                    key: np.stack(values, axis=0).astype(np.float32)
                    for key, values in layer_logs.items()
                }
                for layer_name, layer_logs in self._layer_logs.items()
            },
            "weights_full": {
                layer_name: np.stack(values, axis=0).astype(np.float32)
                for layer_name, values in self._layer_weight_logs.items()
            },
        }


class MinimalElementsLoggerCallback(keras.callbacks.Callback):
    """Track count of Pareto-minimal elements across training epochs."""

    def __init__(self, epoch_stride: int = MINIMAL_ELEMENTS_LOG_EPOCH_STRIDE) -> None:
        super().__init__()
        self.epoch_stride = max(1, int(epoch_stride))
        self.epochs: list[int] = []
        self.total_counts: list[int] = []
        self._per_layer_counts: dict[str, list[int]] = {}
        self._last_epoch_seen = 0

    def on_train_begin(self, logs=None) -> None:
        del logs
        self.epochs = []
        self.total_counts = []
        self._per_layer_counts = {}
        self._last_epoch_seen = 0
        self._snapshot(epoch=0)

    def on_epoch_end(self, epoch, logs=None) -> None:
        del logs
        epoch_num = int(epoch) + 1
        self._last_epoch_seen = epoch_num
        if epoch_num % self.epoch_stride == 0:
            self._snapshot(epoch=epoch_num)

    def on_train_end(self, logs=None) -> None:
        del logs
        if self._last_epoch_seen not in self.epochs:
            self._snapshot(epoch=self._last_epoch_seen)

    def _snapshot(self, epoch: int) -> None:
        self.epochs.append(int(epoch))
        total_minimal = 0
        for layer in iter_my_own_dilation_layers(self.model):
            weights = np.asarray(layer.get_weights()[0], dtype=np.float32)
            flat = weights[0, 0, 0, :, :]
            pareto_mask = is_pareto_efficient(flat.T, return_mask=True)
            n_minimal = int(np.sum(pareto_mask))
            total_minimal += n_minimal
            self._per_layer_counts.setdefault(layer.name, []).append(n_minimal)
        self.total_counts.append(int(total_minimal))

    def export_data(self) -> dict[str, Any]:
        return {
            "epochs": np.asarray(self.epochs, dtype=np.int32),
            "total_minimal_elements": np.asarray(self.total_counts, dtype=np.int32),
            "per_layer_minimal_elements": {
                layer_name: np.asarray(values, dtype=np.int32)
                for layer_name, values in self._per_layer_counts.items()
            },
            "epoch_stride": int(self.epoch_stride),
        }


# %% [markdown]
# ### 6.3 Assembling the callback list
#
# **`build_training_callbacks`** is the single place that decides which callbacks each run gets; extend it when
# you add new training strategies.

# %%
def make_early_stopping_patience_callback(
    patience: int = 25,
    min_delta: float = 1e-5,
    monitor: str = "val_loss",
    *,
    restore_best_weights: bool = True,
    verbose: int = 0,
) -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        verbose=verbose,
    )


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
    cbs: list[keras.callbacks.Callback] = []
    if canonical_architecture_slug(model_slug) in {
        SingleSupErosionsArchitecture.slug,
        TwoLayerSupErosionsArchitecture.slug,
        TwoLayerReceptiveFieldArchitecture.slug,
    }:
        cbs.append(MinimalElementsLoggerCallback())
    if canonical_architecture_slug(model_slug) == SingleSupErosionsArchitecture.slug:
        cbs.append(StructuringElementPairLoggerCallback())
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
            EarlyStopThresholdCallback(val_loss_threshold, monitor=threshold_monitor)
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


# %% [markdown]
# ### 6.4 Checkpoints, run folders, history export
#
# Weight reload and **`histories_to_json_safe`** stay separate from **`fit`** so you can reuse them from alternate
# training entry points.

# %%
def load_best_weights_if_present(model: keras.Model, ck_root: Path, model_slug: str) -> bool:
    p = ck_root / model_slug / "best.weights.h5"
    if not p.is_file():
        return False
    model.load_weights(str(p))
    return True


def new_dated_experiment_dir(parent: Path, prefix: str = "four_pixel") -> Path:
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


# %% [markdown]
# ### 6.5 Single model — compile and fit
#
# Change **optimizer**, **loss**, and **`fit`** kwargs here when experimenting with batching.

# %%
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
    update_rule: str = TRAINING_UPDATE_RULE,
    sparse_mask_min: float = SPARSE_SUBMODEL_MASK_MIN,
    sparse_mask_max: float = SPARSE_SUBMODEL_MASK_MAX,
    sparse_grad_zero_atol: float = SPARSE_GRAD_ZERO_ATOL,
) -> tuple[Any, float]:
    ur = str(update_rule).lower().strip()
    if ur != "standard":
        raise ValueError(
            f"Unsupported update_rule={update_rule!r}; only 'standard' is enabled in this notebook."
        )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=mse_crop_pred_to_true,
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


# %% [markdown]
# ### 6.6 Suite — loop over architectures
#
# Orchestration only: build each model, attach callbacks, call **`train_model`**, collect histories and **val MSE**.

# %%
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
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA,
    early_stopping_restore_best_weights: bool = EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    update_rule: str = TRAINING_UPDATE_RULE,
    sparse_mask_min: float = SPARSE_SUBMODEL_MASK_MIN,
    sparse_mask_max: float = SPARSE_SUBMODEL_MASK_MAX,
    sparse_grad_zero_atol: float = SPARSE_GRAD_ZERO_ATOL,
    k_deactivated_components: int = 0,
) -> dict:
    h_in = int(pack["H"])
    w_in = int(pack["W"])
    x_tr = pack["x_train"]
    x_va = pack["x_val"]
    input_shape = (h_in, w_in, 1)
    arch_list = parse_models_to_train(models_to_train)
    ck = checkpoint_root
    results: dict = {
        "input_shape": input_shape,
        "histories": {},
        "val_mse": {},
        "models": {},
        "weight_pair_logs": {},
        "minimal_elements_logs": {},
        "models_to_train": [c.slug for c in arch_list],
        "checkpoint_root": str(ck) if ck is not None else None,
        "update_rule": str(update_rule),
        "sparse_mask_min": float(sparse_mask_min),
        "sparse_mask_max": float(sparse_mask_max),
        "sparse_grad_zero_atol": float(sparse_grad_zero_atol),
        "k_deactivated_components": int(k_deactivated_components),
    }

    def _cbs(name: str) -> list[keras.callbacks.Callback]:
        return build_training_callbacks(
            ck,
            name,
            mode=callback_mode,
            val_loss_threshold=val_loss_threshold,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            restore_best_weights=early_stopping_restore_best_weights,
        )

    for arch_cls in arch_list:
        slug = arch_cls.slug
        model = arch_cls.build(
            input_shape,
            kernel_size,
            k_deactivated_components=k_deactivated_components,
        )
        disp = arch_cls.display_name
        print("\n" + "=" * 72)
        print(f"Model — before training: {disp} ({slug})")
        print("=" * 72)
        model.summary()
        callbacks = _cbs(slug)
        history, val_mse = train_model(
            model,
            x_tr,
            y_train,
            x_va,
            y_val,
            name=slug,
            ck_root=ck,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            update_rule=update_rule,
            sparse_mask_min=sparse_mask_min,
            sparse_mask_max=sparse_mask_max,
            sparse_grad_zero_atol=sparse_grad_zero_atol,
        )
        pair_logger = next(
            (cb for cb in callbacks if isinstance(cb, StructuringElementPairLoggerCallback)),
            None,
        )
        minimal_logger = next(
            (cb for cb in callbacks if isinstance(cb, MinimalElementsLoggerCallback)),
            None,
        )
        results["histories"][slug] = history
        results["models"][slug] = model
        results["val_mse"][slug] = val_mse
        if pair_logger is not None:
            results["weight_pair_logs"][slug] = pair_logger.export_data()
        if minimal_logger is not None:
            results["minimal_elements_logs"][slug] = minimal_logger.export_data()
        print(f"\n--- After training: {disp} ({slug}) ---")
        print(f"  Validation MSE: {val_mse:.6f}")

    return results


def train_and_eval_k_deactivated_variants_suite(
    pack: dict,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    base_arch_slugs: Sequence[str],
    k_values: Sequence[int],
    kernel_size: tuple[int, int] = KERNEL_SIZE,
    epochs: int = EPOCHS_MAIN,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    verbose: int = 1,
    checkpoint_root: Path | None = None,
    callback_mode: str = TRAINING_CALLBACK_MODE,
    val_loss_threshold: float | None = VAL_LOSS_EARLY_STOP_THRESHOLD,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA,
    early_stopping_restore_best_weights: bool = EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    update_rule: str = TRAINING_UPDATE_RULE,
    sparse_mask_min: float = SPARSE_SUBMODEL_MASK_MIN,
    sparse_mask_max: float = SPARSE_SUBMODEL_MASK_MAX,
    sparse_grad_zero_atol: float = SPARSE_GRAD_ZERO_ATOL,
) -> dict:
    """Train each base architecture for every ``k`` in ``k_values`` in one suite (composite model keys)."""
    h_in = int(pack["H"])
    w_in = int(pack["W"])
    x_tr = pack["x_train"]
    x_va = pack["x_val"]
    input_shape = (h_in, w_in, 1)
    ck = checkpoint_root
    ks = [int(k) for k in k_values]
    bases = [str(s).strip() for s in base_arch_slugs]
    for b in bases:
        if b not in ARCHITECTURE_BY_SLUG:
            raise ValueError(
                f"Unknown architecture slug {b!r}. Use one of: {list(ARCHITECTURE_BY_SLUG.keys())}"
            )
    models_to_train: list[str] = []
    init_variants: list[dict[str, Any]] = []
    for b in bases:
        for k in ks:
            variant = k_deactivated_variant_slug(b, k)
            models_to_train.append(variant)
            init_variants.append(
                {"base_slug": b, "k_deactivated_components": k, "variant_slug": variant}
            )

    results: dict = {
        "input_shape": input_shape,
        "histories": {},
        "val_mse": {},
        "models": {},
        "weight_pair_logs": {},
        "minimal_elements_logs": {},
        "models_to_train": models_to_train,
        "checkpoint_root": str(ck) if ck is not None else None,
        "update_rule": str(update_rule),
        "sparse_mask_min": float(sparse_mask_min),
        "sparse_mask_max": float(sparse_mask_max),
        "sparse_grad_zero_atol": float(sparse_grad_zero_atol),
        "k_deactivated_components": None,
        "initialization_variants": init_variants,
        "protocol": "k_deactivated_initialization_grid",
    }

    def _cbs(name: str) -> list[keras.callbacks.Callback]:
        return build_training_callbacks(
            ck,
            name,
            mode=callback_mode,
            val_loss_threshold=val_loss_threshold,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            restore_best_weights=early_stopping_restore_best_weights,
        )

    for b in bases:
        arch_cls = ARCHITECTURE_BY_SLUG[b]
        for k in ks:
            variant = k_deactivated_variant_slug(b, k)
            model = arch_cls.build(
                input_shape,
                kernel_size,
                k_deactivated_components=k,
            )
            disp = arch_cls.display_name
            print("\n" + "=" * 72)
            print(f"Model — before training: {disp} — k_deact={k} ({variant})")
            print("=" * 72)
            model.summary()
            callbacks = _cbs(variant)
            history, val_mse = train_model(
                model,
                x_tr,
                y_train,
                x_va,
                y_val,
                name=variant,
                ck_root=ck,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks,
                update_rule=update_rule,
                sparse_mask_min=sparse_mask_min,
                sparse_mask_max=sparse_mask_max,
                sparse_grad_zero_atol=sparse_grad_zero_atol,
            )
            pair_logger = next(
                (cb for cb in callbacks if isinstance(cb, StructuringElementPairLoggerCallback)),
                None,
            )
            minimal_logger = next(
                (cb for cb in callbacks if isinstance(cb, MinimalElementsLoggerCallback)),
                None,
            )
            results["histories"][variant] = history
            results["models"][variant] = model
            results["val_mse"][variant] = val_mse
            if pair_logger is not None:
                results["weight_pair_logs"][variant] = pair_logger.export_data()
            if minimal_logger is not None:
                results["minimal_elements_logs"][variant] = minimal_logger.export_data()
            print(f"\n--- After training: {disp} ({variant}) ---")
            print(f"  Validation MSE: {val_mse:.6f}")

    return results


# %% [markdown]
# ## 7–8. Analysis and plots
#
# After training we summarize **learning curves**, **prediction** panels against ground truth, and
# **structuring-element** heatmaps. **Pareto** screening keeps only nondominated learned kernels (coordinate-wise
# on the flattened stencil); large filter banks are split across **paginated** PNGs when needed. Everything below
# uses **NumPy**, **Matplotlib**, and **Keras** only.

# %%
def extract_pareto_filters(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From weights (1,1,1,patch_size,n_filters) return (pareto_filters, mask)."""
    flat = weights[0, 0, 0, :, :]
    filters_t = flat.T.copy()
    mask = is_pareto_efficient(filters_t, return_mask=True)
    return flat[:, mask], mask


def save_weight_pair_logs(suite: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    models = suite.get("models") or {}
    logs_by_model = suite.get("weight_pair_logs") or {}
    for slug, model_logs in logs_by_model.items():
        if canonical_architecture_slug(slug) != SingleSupErosionsArchitecture.slug:
            continue
        model = models.get(slug)
        if model is None:
            continue
        epochs = np.asarray(model_logs.get("epochs", []), dtype=np.int32)
        for layer_name, layer_logs in (model_logs.get("layers") or {}).items():
            payload: dict[str, np.ndarray] = {"epochs": epochs}
            for spec in STRUCTURING_ELEMENT_PAIR_SPECS:
                payload[spec["key"]] = np.asarray(layer_logs[spec["key"]], dtype=np.float32)
            weights_full = (model_logs.get("weights_full") or {}).get(layer_name)
            if weights_full is not None:
                payload["weights_full"] = np.asarray(weights_full, dtype=np.float32)
            out_path = out_dir / f"{slug}_{layer_name}_weight_pair_logs.npz"
            np.savez_compressed(out_path, **payload)
            written.append(out_path)
    return written


def minimal_elements_logs_to_json_safe(logs_by_model: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for slug, logs in (logs_by_model or {}).items():
        epochs = np.asarray(logs.get("epochs", []), dtype=np.int32)
        total = np.asarray(logs.get("total_minimal_elements", []), dtype=np.int32)
        per_layer = logs.get("per_layer_minimal_elements") or {}
        out[slug] = {
            "epochs": [int(x) for x in epochs.tolist()],
            "total_minimal_elements": [int(x) for x in total.tolist()],
            "per_layer_minimal_elements": {
                layer_name: [int(x) for x in np.asarray(vals, dtype=np.int32).tolist()]
                for layer_name, vals in per_layer.items()
            },
            "epoch_stride": int(logs.get("epoch_stride", MINIMAL_ELEMENTS_LOG_EPOCH_STRIDE)),
        }
    return out


def _paginated_save_path(base: Path, n_pages: int, page_index: int) -> Path:
    base = Path(base)
    if n_pages <= 1:
        return base
    return base.parent / f"{base.stem}_p{page_index + 1:02d}{base.suffix}"


def _page_ranges(n_items: int, filters_per_page: int | None) -> list[tuple[int, int]]:
    if filters_per_page is None or filters_per_page <= 0:
        return [(0, n_items)]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n_items:
        end = min(start + filters_per_page, n_items)
        ranges.append((start, end))
        start = end
    return ranges


def plot_structuring_elements(
    weights: np.ndarray,
    kernel_shape: tuple[int, int] = (3, 3),
    n_show: int | None = None,
    rows: int = 15,
    cols: int = 10,
    title: str = "Learned Structuring Elements",
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[Path]:
    flat = weights[0, 0, 0, :, :]
    kernels = flat.T.reshape(-1, *kernel_shape)
    n_total = kernels.shape[0]
    n_show = min(n_show or n_total, n_total)
    filters_per_page = rows * cols
    pages = _page_ranges(n_show, filters_per_page)
    saved: list[Path] = []
    vmin, vmax = -1.1, 1.1
    for page_idx, (start, end) in enumerate(pages):
        n_this = end - start
        rows = (n_this + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        n_cells = rows * cols
        for slot in range(n_this):
            global_i = start + slot
            ax = axes_flat[slot]
            k = kernels[global_i]
            ax.imshow(k, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            for r in range(kernel_shape[0]):
                for c in range(kernel_shape[1]):
                    val = k[r, c]
                    color = "blue" if val > 0 else "red"
                    ax.text(
                        c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8
                    )
            ax.set_title(f"Kernel {global_i}", fontsize=8)
            ax.axis("off")
        for j in range(n_this, n_cells):
            axes_flat[j].axis("off")
        page_title = title
        if len(pages) > 1:
            page_title = f"{title} — page {page_idx + 1} / {len(pages)}"
        plt.suptitle(page_title, fontsize=14, y=1.00)
        plt.tight_layout()
        if save_path:
            out = _paginated_save_path(Path(save_path), len(pages), page_idx)
            plt.savefig(out, bbox_inches="tight", dpi=150)
            saved.append(out)
        if show:
            plt.show()
        else:
            plt.close()
    return saved


def plot_pareto_elements(
    filters: np.ndarray,
    kernel_shape: tuple[int, int] = (3, 3),
    cols: int = 10,
    filters_per_page: int | None = None,
    title: str = "Minimal Structuring Elements",
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[Path]:
    n_filters = filters.shape[1]
    kernels = filters.T.reshape(-1, *kernel_shape)
    pages = _page_ranges(n_filters, filters_per_page)
    saved: list[Path] = []
    vmin, vmax = float(np.min(filters)), float(np.max(filters))
    for page_idx, (start, end) in enumerate(pages):
        n_this = end - start
        rows = (n_this + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        n_cells = rows * cols
        for slot in range(n_this):
            i = start + slot
            ax = axes_flat[slot]
            k = kernels[i]
            ax.imshow(k, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            for r in range(kernel_shape[0]):
                for c in range(kernel_shape[1]):
                    val = k[r, c]
                    color = "blue" if val > 0 else "red"
                    ax.text(
                        c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8
                    )
            ax.axis("off")
        for j in range(n_this, n_cells):
            axes_flat[j].axis("off")
        page_title = title
        if len(pages) > 1:
            page_title = f"{title} — page {page_idx + 1} / {len(pages)}"
        plt.suptitle(page_title, fontsize=14, y=1.00)
        plt.tight_layout()
        if save_path:
            out = _paginated_save_path(Path(save_path), len(pages), page_idx)
            plt.savefig(out, bbox_inches="tight", dpi=150)
            saved.append(out)
        if show:
            plt.show()
        else:
            plt.close()
    return saved


_OVERLAY_COLORS = tuple(plt.get_cmap("tab20").colors)

def _show_fig_if_interactive() -> None:
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def select_weight_pair_snapshot_indices(epochs: np.ndarray) -> list[int]:
    epochs = np.asarray(epochs, dtype=np.int32)
    if epochs.ndim != 1 or len(epochs) == 0:
        return []
    targets = list(WEIGHT_PAIR_SNAPSHOT_EPOCHS)
    selected = [idx for idx, epoch in enumerate(epochs) if int(epoch) in targets]
    last_idx = len(epochs) - 1
    if last_idx not in selected:
        selected.append(last_idx)
    return selected


def plot_structuring_element_pair_snapshot(
    pairs: np.ndarray,
    *,
    pareto_mask: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    expected_point: tuple[float, float],
    epoch_label: str,
    save_path: Path | None = None,
    show: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path | None:
    pairs = np.asarray(pairs, dtype=np.float32)
    if pairs.ndim != 2 or pairs.shape[-1] != 2 or pairs.shape[0] == 0:
        return None
    pareto_mask = np.asarray(pareto_mask, dtype=bool)
    n_filters = pairs.shape[0]
    if pareto_mask.shape != (n_filters,):
        pareto_mask = np.zeros(n_filters, dtype=bool)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.axhline(0.0, color="#d0d0d0", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.8, zorder=0)

    for filt_idx in range(n_filters):
        xy = pairs[filt_idx]
        is_pareto = bool(pareto_mask[filt_idx])
        color = "red" if is_pareto else "blue"
        alpha = 0.9 if is_pareto else 0.35
        zorder = 3 if is_pareto else 1
        ax.scatter(
            xy[0],
            xy[1],
            color=color,
            s=18 if is_pareto else 10,
            alpha=alpha,
            zorder=zorder,
        )

    ax.scatter(
        [expected_point[0]],
        [expected_point[1]],
        color="black",
        marker="*",
        s=140,
        label=f"Expected point {expected_point}",
        zorder=5,
    )
    ax.set_xlim(*(xlim if xlim is not None else (-1.0, 1.0)))
    ax.set_ylim(*(ylim if ylim is not None else (-1.0, 1.0)))
    ax.set_title(f"{title} — epoch {epoch_label}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        _show_fig_if_interactive()
    plt.close(fig)
    return save_path


def plot_logged_weight_pair_snapshot_grid(
    *,
    weight_pair_logs: dict[str, Any] | None,
    model_plots_dir: Path,
    layer_name: str,
    experiment_label: str,
    show: bool = False,
) -> list[Path]:
    if not weight_pair_logs:
        return []
    layer_logs = (weight_pair_logs.get("layers") or {}).get(layer_name)
    if layer_logs is None:
        return []
    layer_weights = (weight_pair_logs.get("weights_full") or {}).get(layer_name)
    if layer_weights is None:
        return []
    epochs = np.asarray(weight_pair_logs.get("epochs", []), dtype=np.int32)
    snapshot_indices = select_weight_pair_snapshot_indices(epochs)
    if not snapshot_indices:
        return []

    def _plot_specs_evolution(specs: Sequence[dict[str, Any]], save_path: Path, *, title_suffix: str) -> Path | None:
        if not specs:
            return None
        n_rows = len(specs)
        n_cols = len(snapshot_indices)
        fig_w = max(2.9 * n_cols, 7.0)
        fig_h = max(2.9 * n_rows, 5.5 if n_rows > 1 else 3.2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
        axes = np.asarray(axes)

        for col_idx, snap_idx in enumerate(snapshot_indices):
            if snap_idx >= layer_weights.shape[0]:
                continue
            epoch_value = int(epochs[snap_idx])
            epoch_label = "last" if snap_idx == len(epochs) - 1 else str(epoch_value)
            pair_values_by_key = {
                spec["key"]: np.asarray(layer_logs[spec["key"]][snap_idx], dtype=np.float32)
                for spec in specs
            }
            pareto_mask = _pair_snapshot_pareto_mask(layer_weights[snap_idx])

            for row_idx, spec in enumerate(specs):
                ax = axes[row_idx, col_idx]
                pairs = pair_values_by_key[spec["key"]]
                if pairs.ndim != 2 or pairs.shape[-1] != 2 or pairs.shape[0] == 0:
                    ax.axis("off")
                    continue
                n_filters = pairs.shape[0]
                mask = np.asarray(pareto_mask, dtype=bool)
                if mask.shape != (n_filters,):
                    mask = np.zeros(n_filters, dtype=bool)

                ax.axhline(0.0, color="#d0d0d0", linewidth=0.8, zorder=0)
                ax.axvline(0.0, color="#d0d0d0", linewidth=0.8, zorder=0)
                ax.scatter(
                    pairs[~mask, 0],
                    pairs[~mask, 1],
                    color="blue",
                    s=10,
                    alpha=0.35,
                    zorder=1,
                    label="non-minimal",
                )
                ax.scatter(
                    pairs[mask, 0],
                    pairs[mask, 1],
                    color="red",
                    s=18,
                    alpha=0.9,
                    zorder=3,
                    label="Pareto minimal",
                )
                ax.scatter(
                    [spec["expected_point"][0]],
                    [spec["expected_point"][1]],
                    color="black",
                    marker="*",
                    s=100,
                    label=f"Expected {spec['expected_point']}",
                    zorder=4,
                )
                ax.grid(True, alpha=0.25)
                ax.tick_params(axis="both", labelsize=7)
                if row_idx == 0:
                    ax.set_title(f"epoch {epoch_label}", fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(spec["ylabel"], fontsize=8)
                if row_idx == n_rows - 1:
                    ax.set_xlabel(spec["xlabel"], fontsize=8)
                if row_idx == 0 and col_idx == n_cols - 1:
                    ax.legend(fontsize=6, loc="upper right")
                ax.set_xlim(*spec.get("xlim", (-1.0, 1.0)))
                ax.set_ylim(*spec.get("ylim", (-1.0, 1.0)))

        fig.suptitle(
            f"{experiment_label} — {layer_name} — {title_suffix}",
            fontsize=11,
        )
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            _show_fig_if_interactive()
        plt.close(fig)
        return save_path if save_path.exists() else None

    layer_weights = np.asarray(layer_weights, dtype=np.float32)
    main_specs = [
        spec
        for spec in STRUCTURING_ELEMENT_PAIR_SPECS
        if spec["key"] != INDEPENDENT_STRUCTURING_ELEMENT_PAIR_KEY
    ]
    independent_specs = [
        spec
        for spec in STRUCTURING_ELEMENT_PAIR_SPECS
        if spec["key"] == INDEPENDENT_STRUCTURING_ELEMENT_PAIR_KEY
    ]

    written: list[Path] = []
    main_path = _plot_specs_evolution(
        main_specs,
        Path(model_plots_dir) / f"{layer_name}_pair_snapshot_evolution_grid.png",
        title_suffix="structuring-element components evolution",
    )
    if main_path is not None:
        written.append(main_path)

    independent_path = _plot_specs_evolution(
        independent_specs,
        Path(model_plots_dir)
        / f"{layer_name}_{INDEPENDENT_STRUCTURING_ELEMENT_PAIR_KEY}_evolution_row.png",
        title_suffix=f"{INDEPENDENT_STRUCTURING_ELEMENT_PAIR_KEY} evolution",
    )
    if independent_path is not None:
        written.append(independent_path)

    return written


def _pair_snapshot_pareto_mask(weights_5d: np.ndarray) -> np.ndarray:
    _, mask = extract_pareto_filters(weights_5d)
    return mask


def _split_weight_values_minimal_vs_rest(weights_5d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights_5d = np.asarray(weights_5d, dtype=np.float32)
    flat = weights_5d[0, 0, 0, :, :]
    _, pareto_mask = extract_pareto_filters(weights_5d)
    minimal_vals = flat[:, pareto_mask].ravel()
    rest_vals = flat[:, ~pareto_mask].ravel()
    return minimal_vals, rest_vals


def plot_logged_weight_value_histograms(
    *,
    weight_pair_logs: dict[str, Any] | None,
    model_plots_dir: Path,
    model_slug: str,
    layer_names: Sequence[str],
    experiment_label: str,
    bins: int = 40,
    show: bool = False,
) -> list[Path]:
    if not weight_pair_logs:
        return []
    epochs = np.asarray(weight_pair_logs.get("epochs", []), dtype=np.int32)
    snapshot_indices = select_weight_pair_snapshot_indices(epochs)
    if not snapshot_indices:
        return []
    layer_weights_by_name = weight_pair_logs.get("weights_full") or {}
    valid_layers = [name for name in layer_names if layer_weights_by_name.get(name) is not None]
    if not valid_layers:
        return []

    n_rows = len(valid_layers)
    n_cols = len(snapshot_indices)
    grid: list[list[tuple[np.ndarray, np.ndarray] | None]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]
    all_values: list[np.ndarray] = []
    for row_idx, layer_name in enumerate(valid_layers):
        layer_weights = np.asarray(layer_weights_by_name[layer_name], dtype=np.float32)
        if layer_weights.ndim != 6:
            continue
        for col_idx, snap_idx in enumerate(snapshot_indices):
            if snap_idx >= layer_weights.shape[0]:
                continue
            minimal_vals, rest_vals = _split_weight_values_minimal_vs_rest(layer_weights[snap_idx])
            grid[row_idx][col_idx] = (minimal_vals, rest_vals)
            all_values.append(np.r_[minimal_vals, rest_vals])
    if not all_values:
        return []

    # Fixed horizontal range [-1, 1] for comparable plots across runs.
    n_bins = int(bins)
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.92
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig_w = max(2.8 * n_cols, 6.5)
    fig_h = max(2.8 * n_rows, 3.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = np.asarray(axes)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    for row_idx, layer_name in enumerate(valid_layers):
        for col_idx, snap_idx in enumerate(snapshot_indices):
            ax = axes[row_idx, col_idx]
            cell = grid[row_idx][col_idx]

            if cell is None:
                ax.axis("off")
                continue

            minimal_vals, rest_vals = cell

            # --- MAIN HISTOGRAM (FULL RANGE) ---
            all_vals = np.r_[minimal_vals, rest_vals]
            xmin, xmax = np.min(all_vals), np.max(all_vals)

            # dynamic bins for full range
            h_min_full, edges_full = np.histogram(minimal_vals, bins=n_bins, range=(xmin, xmax))
            h_rest_full, _ = np.histogram(rest_vals, bins=n_bins, range=(xmin, xmax))

            centers_full = 0.5 * (edges_full[:-1] + edges_full[1:])
            bar_width_full = (edges_full[1] - edges_full[0]) * 0.92

            ax.bar(
                centers_full,
                h_rest_full,
                width=bar_width_full,
                align="center",
                color="blue",
                alpha=0.55,
                label="non-minimal",
            )
            ax.bar(
                centers_full,
                h_min_full,
                width=bar_width_full,
                align="center",
                color="red",
                alpha=0.55,
                label="Pareto minimal",
            )

            # dynamic limits with margin
            margin = 0.05 * (xmax - xmin + 1e-6)
            ax.set_xlim(xmin - margin, xmax + margin)

            # --- TITLES / LABELS ---
            epoch_value = int(epochs[snap_idx])
            epoch_label = "last" if snap_idx == len(epochs) - 1 else str(epoch_value)
            ax.set_title(f"epoch {epoch_label}", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

            if row_idx == n_rows - 1:
                ax.set_xlabel("weight value", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{layer_name}\nfreq.", fontsize=8)
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(fontsize=6, loc="upper right")

            # --- INSET: centered zoom for the typical range [-1, 1] ---
            has_outliers = (xmin < -1.0) or (xmax > 1.0)
            if has_outliers:
                axins = inset_axes(ax, width="46%", height="46%", loc="center")

                h_min_zoom, _ = np.histogram(minimal_vals, bins=n_bins, range=(-1.0, 1.0))
                h_rest_zoom, _ = np.histogram(rest_vals, bins=n_bins, range=(-1.0, 1.0))

                axins.bar(
                    centers,
                    h_rest_zoom,
                    width=bar_width,
                    align="center",
                    color="blue",
                    alpha=0.55,
                )
                axins.bar(
                    centers,
                    h_min_zoom,
                    width=bar_width,
                    align="center",
                    color="red",
                    alpha=0.55,
                )

                axins.set_xlim(-1.0, 1.0)
                # Use the combined envelope so the inset fully covers the visible zoomed bars.
                ymax_zoom = max(
                    float(np.max(h_min_zoom)),
                    float(np.max(h_rest_zoom)),
                    float(np.max(h_min_zoom + h_rest_zoom)),
                )
                axins.set_ylim(0.0, ymax_zoom * 1.12 if ymax_zoom > 0 else 1.0)
                axins.set_title("zoom [-1, 1]", fontsize=6, pad=2)
                axins.tick_params(axis="both", labelsize=6)
                mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.1")

    fig.suptitle(
        f"{experiment_label} — {model_slug} — weight histograms by epoch "
        "(Pareto minimal vs non-minimal; x in [-1, 1])",
        fontsize=11,
    )
    plt.tight_layout()
    save_path = Path(model_plots_dir) / "weight_histogram_evolution_grid.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        _show_fig_if_interactive()
    plt.close(fig)
    return [save_path] if save_path.exists() else []


def plot_logged_weight_pair_snapshots(
    *,
    weight_pair_logs: dict[str, Any] | None,
    model_plots_dir: Path,
    layer_name: str,
    experiment_label: str,
    show: bool = False,
) -> list[Path]:
    if not weight_pair_logs:
        return []
    layer_logs = (weight_pair_logs.get("layers") or {}).get(layer_name)
    if layer_logs is None:
        return []
    layer_weights = (weight_pair_logs.get("weights_full") or {}).get(layer_name)
    if layer_weights is None:
        return []
    model_plots_dir = Path(model_plots_dir)
    pair_dir = model_plots_dir / f"{layer_name}_pair_snapshots"
    pair_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray(weight_pair_logs.get("epochs", []), dtype=np.int32)
    snapshot_indices = select_weight_pair_snapshot_indices(epochs)
    if not snapshot_indices:
        return []
    written: list[Path] = []
    for idx in snapshot_indices:
        epoch_value = int(epochs[idx])
        epoch_label = "last" if idx == len(epochs) - 1 else str(epoch_value)
        pair_values_by_key = {
            spec["key"]: np.asarray(layer_logs[spec["key"]][idx], dtype=np.float32)
            for spec in STRUCTURING_ELEMENT_PAIR_SPECS
        }
        pareto_mask = _pair_snapshot_pareto_mask(
            np.asarray(layer_weights[idx], dtype=np.float32)
        )
        for spec in STRUCTURING_ELEMENT_PAIR_SPECS:
            save_path = pair_dir / f"{spec['key']}_epoch_{epoch_value:04d}.png"
            out = plot_structuring_element_pair_snapshot(
                pair_values_by_key[spec["key"]],
                pareto_mask=pareto_mask,
                title=f"{experiment_label} — {layer_name} — {spec['title']}",
                xlabel=spec["xlabel"],
                ylabel=spec["ylabel"],
                expected_point=spec["expected_point"],
                epoch_label=epoch_label,
                save_path=save_path,
                show=show,
                xlim=spec.get("xlim"),
                ylim=spec.get("ylim"),
            )
            if out is not None and Path(out).exists():
                written.append(Path(out))
    return written


def plot_training_curves_inline(
    history: dict,
    save_path: str | Path,
    title: str,
    *,
    log_scale: bool = False,
) -> None:
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
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both" if log_scale else "major")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_results_overlay(
    results: dict,
    *,
    log_scale: bool = False,
    mode: str = "val",
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    mode_norm = mode.strip().lower()
    if mode_norm not in ("val", "train_val"):
        raise ValueError("mode must be 'val' or 'train_val'")
    if title is None:
        title = "Validation loss" if mode_norm == "val" else "Train vs val loss"
    colors = _OVERLAY_COLORS
    histories = results.get("histories") or {}
    items = list(histories.items())
    if not items:
        return
    if mode_norm == "val":
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (name, hist) in enumerate(items):
            hist_dict = hist.history if hasattr(hist, "history") else hist
            vl = hist_dict.get("val_loss", [])
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
            hist_dict = hist.history if hasattr(hist, "history") else hist
            loss = hist_dict.get("loss", [])
            vl = hist_dict.get("val_loss", [])
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
                    ax0.plot(range(1, len(loss) + 1), loss, color=c, label=name, linewidth=1.2)
                if vl:
                    ax1.plot(range(1, len(vl) + 1), vl, color=c, label=name, linewidth=1.2)
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


def save_per_model_curves(results: dict, out_dir: Path | None) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    histories = results.get("histories") or {}
    for name, hist in histories.items():
        model_dir = out_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        hdict = hist.history if hasattr(hist, "history") else hist
        plot_training_curves_inline(
            hdict, model_dir / "loss.png", title=f"{name} — training", log_scale=False
        )
        plot_training_curves_inline(
            hdict,
            model_dir / "loss_log.png",
            title=f"{name} — training (log)",
            log_scale=True,
        )


def plot_training_curves_overlay(
    results: dict,
    *,
    log_scale: bool = False,
    save_path: Path | None = None,
    title: str | None = None,
    target_label: str = "four-pixel average target",
    model_slugs: Sequence[str] | None = None,
) -> None:
    keys = list(model_slugs) if model_slugs is not None else list(PREFERRED_MODEL_ORDER)
    hists: list[tuple[str, dict]] = []
    labels_short: list[str] = []
    for k in keys:
        h = results["histories"].get(k)
        if h is None:
            continue
        hi = h.history if hasattr(h, "history") else h
        hists.append((k, hi))
        labels_short.append(
            ARCHITECTURE_BY_SLUG[k].display_name if k in ARCHITECTURE_BY_SLUG else k
        )
    if not hists:
        return
    colors = _OVERLAY_COLORS
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


def plot_minimal_elements_overlay(
    results: dict,
    *,
    save_path: Path | None = None,
    title: str | None = None,
    model_slugs=None,
    epoch_stride: int = MINIMAL_ELEMENTS_LOG_EPOCH_STRIDE,
) -> None:
    logs_by_model = results.get("minimal_elements_logs") or {}
    if not logs_by_model:
        return
    keys = list(model_slugs) if model_slugs is not None else list(PREFERRED_MODEL_ORDER)
    keys += [k for k in logs_by_model.keys() if k not in keys]
    stride = max(1, int(epoch_stride))
    marker_cycle = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
    colors = _OVERLAY_COLORS
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for i, slug in enumerate(keys):
        logs = logs_by_model.get(slug)
        if not logs:
            continue
        epochs = np.asarray(logs.get("epochs", []), dtype=np.int32)
        totals = np.asarray(logs.get("total_minimal_elements", []), dtype=np.int32)
        if len(epochs) == 0 or len(totals) == 0:
            continue
        n = min(len(epochs), len(totals))
        epochs = epochs[:n]
        totals = totals[:n]
        keep = (epochs % stride) == 0
        keep[-1] = True
        epochs_plot = epochs[keep]
        totals_plot = totals[keep]

        if len(epochs_plot) == 0:
            continue
        label = ARCHITECTURE_BY_SLUG[slug].display_name if slug in ARCHITECTURE_BY_SLUG else slug
        ax.plot(
            epochs_plot,
            totals_plot,
            color=colors[i % len(colors)],
            marker=marker_cycle[i % len(marker_cycle)],
            linewidth=1.5,
            markersize=5,
            label=label,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Minimal Elements")
    ax.set_title(title or "Minimal elements over training")
    # put grid behind everything
    ax.set_axisbelow(True)
    # major grid (----)
    ax.grid(
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )
    # minor ticks (needed for dotted grid)
    ax.minorticks_on()
    # minor grid (......)
    ax.grid(
        which="minor",
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
    )
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    # replace this:
    _show_fig_if_interactive()
    plt.close(fig)

def plot_prediction_row(
    pack: dict,
    results: dict,
    y_val: np.ndarray,
    n_samples: int = 3,
    save_path: Path | None = None,
) -> None:
    x_va = pack["x_val"]
    n = min(n_samples, len(x_va))
    model_items = list(results["models"].items())
    ncols = 2 + len(model_items)
    fig, axes = plt.subplots(n, ncols, figsize=(2.2 * ncols, 2.4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for r in range(n):
        axes[r, 0].imshow(x_va[r, :, :, 0], cmap="bone", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"#{r}", rotation=0, labelpad=12)
        vmin = float(np.min(y_val[r]))
        vmax = float(np.max(y_val[r]))
        axes[r, 1].imshow(y_val[r, :, :, 0], cmap="bone", vmin=vmin, vmax=vmax)
        if r == 0:
            axes[r, 0].set_title("Input")
            axes[r, 1].set_title("Target")
        for j, (name, model) in enumerate(model_items):
            p = model.predict(x_va[r : r + 1], verbose=0)
            p = crop_pred_to_y(np.asarray(p), y_val[r : r + 1])
            span = max(vmax - vmin, 1e-6)
            axes[r, j + 2].imshow(
                p[0, :, :, 0], cmap="bone", vmin=vmin - 0.05 * span, vmax=vmax + 0.05 * span
            )
            if r == 0:
                axes[r, j + 2].set_title(name[:14], fontsize=8)
        for c in range(ncols):
            axes[r, c].axis("off")
    plt.suptitle("Validation: input | target | models", y=1.02)
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
    n_samples: int = 4,
    save_path: Path | None = None,
    suptitle: str | None = None,
    model_slugs: Sequence[str] | None = None,
) -> None:
    models_dict = results.get("models") or {}
    slugs = list(model_slugs) if model_slugs is not None else comparison_plot_slugs(results)
    if not slugs:
        return
    x_va = pack["x_val"]
    n = min(n_samples, len(x_va), len(y_val))
    if n <= 0:
        return
    x_s = x_va[:n]
    y_s = y_val[:n]
    preds: list[np.ndarray] = []
    for slug in slugs:
        m = models_dict[slug]
        p = m.predict(x_s, verbose=0)
        preds.append(crop_pred_to_y(np.asarray(p), y_s))
    ncols = 2 + len(slugs)
    fig, axes = plt.subplots(n, ncols, figsize=(max(14, 2.3 * ncols), 2.8 * n), squeeze=False)
    pred_labels = [
        f"Pred — {ARCHITECTURE_BY_SLUG[s].display_name if s in ARCHITECTURE_BY_SLUG else s}"
        for s in slugs
    ]
    col_titles = ["Input", "Target (4-px average)"] + pred_labels
    for r in range(n):
        vmin_gt = float(np.min(y_s[r]))
        vmax_gt = float(np.max(y_s[r]))
        span = max(vmax_gt - vmin_gt, 1e-6)
        vmin_p = vmin_gt - 0.05 * span
        vmax_p = vmax_gt + 0.05 * span
        axes[r, 0].imshow(x_s[r, :, :, 0], cmap="bone", vmin=0, vmax=1)
        axes[r, 1].imshow(y_s[r, :, :, 0], cmap="bone", vmin=vmin_gt, vmax=vmax_gt)
        for j, pred in enumerate(preds):
            axes[r, 2 + j].imshow(pred[r, :, :, 0], cmap="bone", vmin=vmin_p, vmax=vmax_p)
        if r == 0:
            for c in range(ncols):
                axes[0, c].set_title(col_titles[c], fontsize=10)
        for c in range(ncols):
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(f"#{r}", fontsize=10, rotation=0, labelpad=16)
    plt.suptitle(suptitle or "Four-pixel average: input, target, predictions", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    _show_fig_if_interactive()
    plt.close(fig)


def plot_combiner_pareto_pairs(
    model: keras.Model,
    *,
    save_path: Path | None,
    title: str,
    show: bool = False,
) -> list[Path]:
    w1, w2 = model.get_layer("SupErosions_3").get_weights()
    paired = np.column_stack([w1, w2]).astype(np.float32)
    mask = is_pareto_efficient(paired.copy(), return_mask=True)
    pairs = paired[mask]
    if len(pairs) == 0:
        return []
    n = len(pairs)
    cols = min(10, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.atleast_2d(axes)
    vmin, vmax = float(np.min(pairs)), float(np.max(pairs))
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            bar = pairs[i].reshape(1, 2)
            ax.imshow(bar, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
            for j in range(2):
                val = pairs[i, j]
                color = "blue" if val > 0 else "red"
                ax.text(j, 0, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.axis("off")
    plt.suptitle(f"{title} — Pareto pairs (w1, w2) — n={n}", fontsize=12)
    plt.tight_layout()
    saved: list[Path] = []
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        saved.append(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved


def plot_suite_structuring_elements(
    suite: dict,
    plots_dir: Path,
    *,
    experiment_label: str = "Four-pixel average",
    show: bool = False,
) -> list[Path]:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    kh, kw = KERNEL_SIZE
    models = suite.get("models") or {}
    weight_pair_logs = suite.get("weight_pair_logs") or {}

    for slug, m in models.items():
        canon = canonical_architecture_slug(slug)
        model_plots_dir = plots_dir / slug
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        if canon == SingleSupErosionsArchitecture.slug:
            w = m.get_layer("Erosion").get_weights()[0]
            written.extend(
                plot_structuring_elements(
                    w,
                    kernel_shape=(kh, kw),
                    n_show=min(64, N_EROSIONS_SINGLE),
                    title=f"{experiment_label} — {slug} — all structuring elements",
                    save_path=model_plots_dir / "all_kernels.png",
                    show=show,
                )
            )
            written.extend(
                plot_logged_weight_value_histograms(
                    weight_pair_logs=weight_pair_logs.get(slug),
                    model_plots_dir=model_plots_dir,
                    model_slug=slug,
                    layer_names=("Erosion",),
                    experiment_label=experiment_label,
                    show=show,
                )
            )
            written.extend(
                plot_logged_weight_pair_snapshot_grid(
                    weight_pair_logs=weight_pair_logs.get(slug),
                    model_plots_dir=model_plots_dir,
                    layer_name="Erosion",
                    experiment_label=experiment_label,
                    show=show,
                )
            )
            pareto_w, _ = extract_pareto_filters(w)
            written.extend(
                plot_pareto_elements(
                    pareto_w,
                    kernel_shape=(kh, kw),
                    title=f"{experiment_label} — {slug} — Pareto-minimal elements",
                    save_path=model_plots_dir / "pareto_kernels.png",
                    show=show,
                )
            )
        elif canon in (
            TwoLayerSupErosionsArchitecture.slug,
            TwoLayerReceptiveFieldArchitecture.slug,
        ):
            for _branch, layer_name in ((1, "Erosions1"), (2, "Erosions2")):
                w = m.get_layer(layer_name).get_weights()[0]
                written.extend(
                    plot_structuring_elements(
                        w,
                        kernel_shape=(kh, kw),
                        n_show=min(64, w.shape[-1]),
                        title=f"{experiment_label} — {slug} — {layer_name} (all)",
                        save_path=model_plots_dir / f"{layer_name}_all.png",
                        show=show,
                    )
                )
                pareto_w, _ = extract_pareto_filters(w)
                written.extend(
                    plot_pareto_elements(
                        pareto_w,
                        kernel_shape=(kh, kw),
                        title=f"{experiment_label} — {slug} — {layer_name} (Pareto)",
                        save_path=model_plots_dir / f"{layer_name}_pareto.png",
                        show=show,
                    )
                )
            written.extend(
                plot_logged_weight_value_histograms(
                    weight_pair_logs=weight_pair_logs.get(slug),
                    model_plots_dir=model_plots_dir,
                    model_slug=slug,
                    layer_names=("Erosions1", "Erosions2"),
                    experiment_label=experiment_label,
                    show=show,
                )
            )
            for layer_name in ("Erosions1", "Erosions2"):
                written.extend(
                    plot_logged_weight_pair_snapshot_grid(
                        weight_pair_logs=weight_pair_logs.get(slug),
                        model_plots_dir=model_plots_dir,
                        layer_name=layer_name,
                        experiment_label=experiment_label,
                        show=show,
                    )
                )
            written.extend(
                plot_combiner_pareto_pairs(
                    m,
                    save_path=model_plots_dir / "SupErosions_3_pareto_pairs.png",
                    title=f"{experiment_label} — {slug}",
                    show=show,
                )
            )
    return [p for p in written if Path(p).exists()]


def run_comparison_plots(
    suite: dict,
    pack: dict,
    y_val: np.ndarray,
    plots_dir: Path,
    *,
    target_label: str,
    show: bool = True,
    n_pred_samples: int = 4,
    training_overlay_slugs: Sequence[str] | None = None,
    prediction_grid_slugs: Sequence[str] | None = None,
) -> list[Path]:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    comp = comparison_plot_slugs(suite)
    overlay_slugs = (
        list(training_overlay_slugs)
        if training_overlay_slugs is not None
        else comp
    )

    if overlay_slugs:
        plot_training_curves_overlay(
            suite,
            log_scale=False,
            save_path=plots_dir / "four_pixel_three_way_overlay.png",
            target_label=target_label,
            model_slugs=overlay_slugs,
        )
        saved.append(plots_dir / "four_pixel_three_way_overlay.png")
        plot_training_curves_overlay(
            suite,
            log_scale=True,
            save_path=plots_dir / "four_pixel_three_way_overlay_log.png",
            target_label=target_label,
            model_slugs=overlay_slugs,
        )
        saved.append(plots_dir / "four_pixel_three_way_overlay_log.png")
        plot_minimal_elements_overlay(
            suite,
            save_path=plots_dir / "minimal_elements_overlay.png",
            title=f"{target_label} — minimal elements over training",
            model_slugs=overlay_slugs,
            epoch_stride=MINIMAL_ELEMENTS_LOG_EPOCH_STRIDE,
        )
        saved.append(plots_dir / "minimal_elements_overlay.png")

    plot_results_overlay(
        suite,
        log_scale=False,
        mode="val",
        title="Validation loss — all models",
        save_path=plots_dir / "val_loss_overlay.png",
    )
    saved.append(plots_dir / "val_loss_overlay.png")
    plot_results_overlay(
        suite,
        log_scale=True,
        mode="val",
        title="Validation loss — all models (log)",
        save_path=plots_dir / "val_loss_overlay_log.png",
    )
    saved.append(plots_dir / "val_loss_overlay_log.png")
    plot_results_overlay(
        suite,
        log_scale=False,
        mode="train_val",
        title="Train / val — all models",
        save_path=plots_dir / "train_val_overlay.png",
    )
    saved.append(plots_dir / "train_val_overlay.png")
    plot_results_overlay(
        suite,
        log_scale=True,
        mode="train_val",
        title="Train / val — all models (log)",
        save_path=plots_dir / "train_val_overlay_log.png",
    )
    saved.append(plots_dir / "train_val_overlay_log.png")

    save_per_model_curves(suite, plots_dir)
    for name in (suite.get("histories") or {}):
        saved.append(plots_dir / name / "loss.png")
        saved.append(plots_dir / name / "loss_log.png")

    models = suite.get("models") or {}
    grid_slugs_raw = (
        list(prediction_grid_slugs) if prediction_grid_slugs is not None else comp
    )
    grid_slugs = [s for s in grid_slugs_raw if s in models]
    if grid_slugs:
        plot_model_predictions_grid(
            pack,
            suite,
            y_val,
            n_samples=n_pred_samples,
            save_path=plots_dir / "four_pixel_prediction_three_way.png",
            suptitle=f"{target_label}: input, target, predictions",
            model_slugs=grid_slugs,
        )
        saved.append(plots_dir / "four_pixel_prediction_three_way.png")

    if models:
        plot_prediction_row(
            pack,
            suite,
            y_val,
            n_samples=min(3, len(y_val)),
            save_path=plots_dir / "prediction_comparison_all_models.png",
        )
        saved.append(plots_dir / "prediction_comparison_all_models.png")

    se_paths = plot_suite_structuring_elements(
        suite,
        plots_dir,
        experiment_label=target_label,
        show=show,
    )
    saved.extend(se_paths)
    return [p for p in saved if Path(p).exists()]


# %% [markdown]
# ## 9. Interpretation
#
# - **Depth:** Two-layer models build two parallel sup-erosion fields, then the combiner applies an outer
#   $\max_m\min$ over shifted copies. The receptive-field variant forces the two banks to emphasize different
#   stencil entries by masking weights.
# - **Pareto plots:** Structuring elements that are dominated (entry-wise larger) than another filter carry no
#   extra information; the Pareto front is the useful summary of what the network actually uses.
# - **Training:** These networks are sensitive to initialization and scale. **`VAL_LOSS_EARLY_STOP_THRESHOLD`**
#   stops early when **MSE** on $[0,1]$ targets drops near $255^{-2}$, which is a practical "good enough" scale
#   for this pixel resolution.
# - **Paper-style statistics:** Use **`run_experiment_repeated`** and **`repetitions_summary.json`** for mean and
#   standard deviation of validation error and for the **success rate** at threshold $Q$.

# %% [markdown]
# ## 10. Experiment driver
#
# **`run_experiment`** loads data and targets, calls **`train_and_eval_suite`**, writes **`metrics.json`** and
# **`metadata.json`**, renders the full **plot** suite under **`plots/`**, and keeps **weight checkpoints** under
# **`checkpoints/`** inside a timestamped run folder.
#
# **`run_experiment_repeated`** runs the same pipeline **`N_TRAINING_REPETITIONS`** times with a **fixed noisy
# dataset** (**`DATA_LOAD_SEED`**) and a fresh **Keras** RNG seed each time (**`RANDOM_SEED`** +
# **`rep`** $\times$ **`TRAINING_SEED_STEP`**). Each repetition is stored under **`rep_00/`**, **`rep_01/`**, … with
# its own checkpoints and plots; **`repetitions_summary.json`** at the run root reports **mean / std** of
# **val_mse** and the **success rate** (fraction of reps with **val_mse** $< Q$).

# %%
def build_experiment_metadata(
    run_dir: Path, pack: dict, suite: dict, *, target_meta: dict[str, Any]
) -> dict:
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "dataset": pack.get("dataset"),
        "img_size": [int(pack["H"]), int(pack["W"])],
        "hyperparameters": {
            "epochs": EPOCHS_MAIN,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "kernel_size": list(KERNEL_SIZE),
            "n_erosions_single": N_EROSIONS_SINGLE,
            "n_erosions_tl1": N_EROSIONS_TL1,
            "n_erosions_tl2": N_EROSIONS_TL2,
            "n_erosions_tl3": N_EROSIONS_TL3,
            "init_center": INIT_CENTER,
            "init_half_width": INIT_HALF_WIDTH,
            "init_min": INIT_MIN,
            "init_max": INIT_MAX,
            "data_load_seed": DATA_LOAD_SEED,
            "n_training_repetitions_default": N_TRAINING_REPETITIONS,
            "training_seed_step": TRAINING_SEED_STEP,
            "val_loss_threshold": VAL_LOSS_EARLY_STOP_THRESHOLD,
            "training_callback_mode": TRAINING_CALLBACK_MODE,
            "noise_sigma": NOISE_SIGMA,
            "dataset_size": DATASET_SIZE,
            "rf_block1_inactive": RF_BLOCK1_INACTIVE,
            "rf_block2_inactive": RF_BLOCK2_INACTIVE,
            "training_update_rule": suite.get("update_rule", TRAINING_UPDATE_RULE),
            "sparse_submodel_mask_range": [
                suite.get("sparse_mask_min", SPARSE_SUBMODEL_MASK_MIN),
                suite.get("sparse_mask_max", SPARSE_SUBMODEL_MASK_MAX),
            ],
            "sparse_grad_zero_atol": suite.get(
                "sparse_grad_zero_atol", SPARSE_GRAD_ZERO_ATOL
            ),
            "k_deactivated_components": suite.get("k_deactivated_components"),
            "initialization_variants": suite.get("initialization_variants"),
            "suite_protocol": suite.get("protocol"),
        },
        "checkpoint_root": suite.get("checkpoint_root"),
        "val_mse": dict(suite["val_mse"]),
        "models_trained": list(suite["histories"].keys()),
        "weight_pair_logs_dir": suite.get("weight_pair_logs_dir"),
        "weight_pair_log_files": list(suite.get("weight_pair_log_files", [])),
    }
    meta.update(target_meta)
    meta["keras_version"] = keras.__version__
    return meta


def write_metrics_and_metadata(
    run_dir: Path, pack: dict, suite: dict, *, target_meta: dict[str, Any]
) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    meta = build_experiment_metadata(run_dir, pack, suite, target_meta=target_meta)
    metrics = {
        "val_mse": dict(suite["val_mse"]),
        "histories": histories_to_json_safe(suite["histories"]),
        "minimal_elements_logs": minimal_elements_logs_to_json_safe(
            suite.get("minimal_elements_logs", {})
        ),
    }
    mp = run_dir / "metrics.json"
    with open(mp, "w") as f:
        json.dump(metrics, f, indent=2)
    metap = run_dir / "metadata.json"
    with open(metap, "w") as f:
        json.dump(meta, f, indent=2)
    return mp, metap


def print_experiment_summary(run_dir: Path, suite: dict) -> None:
    print("\n" + "=" * 72)
    print("Four-pixel average experiment — summary")
    print("=" * 72)
    print(f"Run directory: {run_dir.resolve()}")
    for slug in suite.get("models_to_train", []):
        print(f"  {slug}: val_mse = {suite['val_mse'].get(slug, float('nan')):.6f}")
    print("=" * 72 + "\n")


def run_experiment(
    out_dir: Path | str | None = None,
    *,
    models_to_train: Sequence[str] | None = None,
    notebook: bool = False,
    dated_subdir: bool = True,
    update_rule: str = TRAINING_UPDATE_RULE,
    sparse_mask_min: float = SPARSE_SUBMODEL_MASK_MIN,
    sparse_mask_max: float = SPARSE_SUBMODEL_MASK_MAX,
    sparse_grad_zero_atol: float = SPARSE_GRAD_ZERO_ATOL,
    k_deactivated_components: int = 0,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")

    parent = Path(out_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f"exp_{EXPERIMENT_TARGET_SLUG}"
    if dated_subdir:
        run_dir = new_dated_experiment_dir(parent, prefix)
    else:
        run_dir = parent
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    plots_dir = run_dir / "plots"
    ck_root = run_dir / "checkpoints"

    pack = load_fashion_mnist_four_pixel_pack(seed=DATA_LOAD_SEED)
    y_train, y_val, _y_test, target_meta = compute_experiment_targets(pack)

    print("Target:", target_meta["target_key"], "—", target_meta["target_label"])
    print("Models:", [c.slug for c in parse_models_to_train(models_to_train)])
    print("Shapes: x_train", pack["x_train"].shape, "y_train", y_train.shape)
    print("Experiment run directory:", run_dir)

    suite = train_and_eval_suite(
        pack,
        y_train,
        y_val,
        epochs=EPOCHS_MAIN,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        verbose=1,
        checkpoint_root=ck_root,
        models_to_train=models_to_train,
        update_rule=update_rule,
        sparse_mask_min=sparse_mask_min,
        sparse_mask_max=sparse_mask_max,
        sparse_grad_zero_atol=sparse_grad_zero_atol,
        k_deactivated_components=k_deactivated_components,
    )
    suite["experiment_run_dir"] = str(run_dir.resolve())
    weight_pair_log_files = save_weight_pair_logs(suite, run_dir / "weight_pair_logs")
    suite["weight_pair_logs_dir"] = str((run_dir / "weight_pair_logs").resolve())
    suite["weight_pair_log_files"] = [str(p.relative_to(run_dir)) for p in weight_pair_log_files]
    write_metrics_and_metadata(run_dir, pack, suite, target_meta=target_meta)
    run_comparison_plots(
        suite,
        pack,
        y_val,
        plots_dir,
        target_label=target_meta["target_label"],
        show=False,
        n_pred_samples=min(4, len(y_val)),
    )
    print_experiment_summary(run_dir, suite)

    if notebook:
        try:
            from IPython.display import Image, Markdown, display

            display(Markdown(f"**Run:** `{run_dir}`"))
            display(Markdown(f"**Figures:** `{plots_dir}`"))
            for p in sorted(plots_dir.glob("*.png")):
                display(Image(filename=str(p)))
        except ImportError:
            pass

    return suite


def run_experiment_repeated(
    out_dir: Path | str | None = None,
    *,
    n_repetitions: int | None = None,
    data_seed: int = DATA_LOAD_SEED,
    training_seed_base: int = RANDOM_SEED,
    training_seed_step: int = TRAINING_SEED_STEP,
    models_to_train: Sequence[str] | None = None,
    notebook: bool = False,
    dated_subdir: bool = True,
    update_rule: str = TRAINING_UPDATE_RULE,
    sparse_mask_min: float = SPARSE_SUBMODEL_MASK_MIN,
    sparse_mask_max: float = SPARSE_SUBMODEL_MASK_MAX,
    sparse_grad_zero_atol: float = SPARSE_GRAD_ZERO_ATOL,
    k_deactivated_components: int = 0,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")

    n = int(N_TRAINING_REPETITIONS if n_repetitions is None else n_repetitions)
    if n < 1:
        raise ValueError("n_repetitions must be >= 1")

    parent = Path(out_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f"exp_{EXPERIMENT_TARGET_SLUG}_repeated"
    if dated_subdir:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        run_dir = parent / f"{prefix}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = parent
        run_dir.mkdir(parents=True, exist_ok=True)

    pack = load_fashion_mnist_four_pixel_pack(seed=data_seed)
    y_train, y_val, _y_test, target_meta = compute_experiment_targets(pack)
    q_thresh = float(VAL_LOSS_EARLY_STOP_THRESHOLD)

    per_rep: list[dict[str, Any]] = []
    last_suite: dict[str, Any] = {}

    for rep in range(n):
        rep_tag = f"rep_{rep:02d}"
        rep_dir = run_dir / rep_tag
        plots_dir = rep_dir / "plots"
        ck_root = rep_dir / "checkpoints"
        plots_dir.mkdir(parents=True, exist_ok=True)
        ck_root.mkdir(parents=True, exist_ok=True)

        rep_training_seed = int(training_seed_base + rep * training_seed_step)
        keras.utils.set_random_seed(rep_training_seed)

        print(
            f"\n{'=' * 72}\nRepetition {rep + 1}/{n}  (training seed={rep_training_seed})\n{'=' * 72}"
        )
        suite = train_and_eval_suite(
            pack,
            y_train,
            y_val,
            epochs=EPOCHS_MAIN,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            verbose=1,
            checkpoint_root=ck_root,
            models_to_train=models_to_train,
            update_rule=update_rule,
            sparse_mask_min=sparse_mask_min,
            sparse_mask_max=sparse_mask_max,
            sparse_grad_zero_atol=sparse_grad_zero_atol,
            k_deactivated_components=k_deactivated_components,
        )
        suite["experiment_run_dir"] = str(rep_dir.resolve())
        suite["repetition_index"] = rep
        suite["training_seed"] = rep_training_seed
        weight_pair_log_files = save_weight_pair_logs(suite, rep_dir / "weight_pair_logs")
        suite["weight_pair_logs_dir"] = str((rep_dir / "weight_pair_logs").resolve())
        suite["weight_pair_log_files"] = [str(p.relative_to(rep_dir)) for p in weight_pair_log_files]

        write_metrics_and_metadata(rep_dir, pack, suite, target_meta=target_meta)
        run_comparison_plots(
            suite,
            pack,
            y_val,
            plots_dir,
            target_label=target_meta["target_label"],
            show=False,
            n_pred_samples=min(4, len(y_val)),
        )

        val_mse = dict(suite["val_mse"])
        success = {slug: float(val_mse[slug]) < q_thresh for slug in val_mse}
        per_rep.append(
            {
                "repetition": rep,
                "training_seed": rep_training_seed,
                "val_mse": {k: float(v) for k, v in val_mse.items()},
                "success": success,
            }
        )
        last_suite = suite

    slugs = list(per_rep[0]["val_mse"].keys())
    aggregate: dict[str, Any] = {}
    for slug in slugs:
        vals = np.array([float(r["val_mse"][slug]) for r in per_rep], dtype=np.float64)
        successes = sum(1 for r in per_rep if r["success"][slug])
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        aggregate[slug] = {
            "mean_val_mse": float(vals.mean()),
            "std_val_mse": std,
            "success_rate": successes / n,
            "n_successes": int(successes),
        }

    summary: dict[str, Any] = {
        "protocol": "four_pixel_average_repeated",
        "n_repetitions": n,
        "Q": q_thresh,
        "data_seed": int(data_seed),
        "training_seed_base": int(training_seed_base),
        "training_seed_step": int(training_seed_step),
        "per_repetition": per_rep,
        "aggregate": aggregate,
    }
    summary_path = run_dir / "repetitions_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Repeated protocol run directory: {run_dir.resolve()}")
    print(f"Summary: {summary_path}")
    for slug in slugs:
        a = aggregate[slug]
        print(
            f"  {slug}: mean val_mse={a['mean_val_mse']:.6f} ± {a['std_val_mse']:.6f}, "
            f"success_rate={a['success_rate']:.2f}"
        )
    print("=" * 72 + "\n")

    if notebook:
        try:
            from IPython.display import Markdown, display

            display(Markdown(f"**Repeated run:** `{run_dir}`"))
            display(Markdown(f"**Summary:** `{summary_path}`"))
        except ImportError:
            pass

    return {
        "run_dir": str(run_dir.resolve()),
        "summary_path": str(summary_path),
        "summary": summary,
        "last_suite": last_suite,
    }


# %%
def run_k_deactivated_initialization_experiment(
    out_dir: Path | str | None = None,
    *,
    notebook: bool = False,
    dated_subdir: bool = True,
    k_values: Sequence[int] | None = None,
    update_rule: str = TRAINING_UPDATE_RULE,
    sparse_mask_min: float = SPARSE_SUBMODEL_MASK_MIN,
    sparse_mask_max: float = SPARSE_SUBMODEL_MASK_MAX,
    sparse_grad_zero_atol: float = SPARSE_GRAD_ZERO_ATOL,
) -> dict[str, Any]:
    """
    Single experiment run (same layout as ``run_experiment``): one ``run_dir``,
    data loaded once, then **10** trainings — each base architecture with
    ``k_deactivated_components`` in ``{0,1,2,3,4}``. Model keys are
    ``{base}__kd{k}`` (e.g. ``single_sup__kd0`` is normal init).

    Combined plots (e.g. ``val_loss_overlay.png``) include **all** variants.
    """
    import matplotlib

    matplotlib.use("Agg")

    parent = Path(out_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f"exp_{EXPERIMENT_TARGET_SLUG}_k_deactivated_grid"
    if dated_subdir:
        run_dir = new_dated_experiment_dir(parent, prefix)
    else:
        run_dir = parent
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    plots_dir = run_dir / "plots"
    ck_root = run_dir / "checkpoints"

    arch_slugs = [
        SingleSupErosionsArchitecture.slug,
        TwoLayerSupErosionsArchitecture.slug,
    ]
    ks = [0, 1, 2, 3, 4] if k_values is None else [int(x) for x in k_values]

    pack = load_fashion_mnist_four_pixel_pack(seed=DATA_LOAD_SEED)
    y_train, y_val, _y_test, target_meta = compute_experiment_targets(pack)

    print("Target:", target_meta["target_key"], "—", target_meta["target_label"])
    print("K-deactivated grid: architectures", arch_slugs, "k_values", ks)
    print("Shapes: x_train", pack["x_train"].shape, "y_train", y_train.shape)
    print("Experiment run directory:", run_dir)

    suite = train_and_eval_k_deactivated_variants_suite(
        pack,
        y_train,
        y_val,
        base_arch_slugs=arch_slugs,
        k_values=ks,
        epochs=EPOCHS_MAIN,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        verbose=1,
        checkpoint_root=ck_root,
        update_rule=update_rule,
        sparse_mask_min=sparse_mask_min,
        sparse_mask_max=sparse_mask_max,
        sparse_grad_zero_atol=sparse_grad_zero_atol,
    )
    suite["experiment_run_dir"] = str(run_dir.resolve())
    variant_slugs = list(suite["models_to_train"])

    weight_pair_log_files = save_weight_pair_logs(suite, run_dir / "weight_pair_logs")
    suite["weight_pair_logs_dir"] = str((run_dir / "weight_pair_logs").resolve())
    suite["weight_pair_log_files"] = [str(p.relative_to(run_dir)) for p in weight_pair_log_files]
    write_metrics_and_metadata(run_dir, pack, suite, target_meta=target_meta)
    run_comparison_plots(
        suite,
        pack,
        y_val,
        plots_dir,
        target_label=target_meta["target_label"],
        show=False,
        n_pred_samples=min(4, len(y_val)),
        training_overlay_slugs=variant_slugs,
        prediction_grid_slugs=variant_slugs,
    )
    print_experiment_summary(run_dir, suite)

    summary = {
        "protocol": suite.get("protocol", "k_deactivated_initialization_grid"),
        "run_dir": str(run_dir.resolve()),
        "architectures": arch_slugs,
        "k_values": ks,
        "variant_slugs": variant_slugs,
        "val_mse": {k: float(v) for k, v in suite.get("val_mse", {}).items()},
    }
    summary_path = run_dir / "k_deactivated_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if notebook:
        try:
            from IPython.display import Image, Markdown, display

            display(Markdown(f"**K-deactivated run:** `{run_dir}`"))
            display(Markdown(f"**Summary:** `{summary_path}`"))
            display(Markdown(f"**Figures:** `{plots_dir}`"))
            for p in sorted(plots_dir.glob("*.png")):
                display(Image(filename=str(p)))
        except ImportError:
            pass

    return {
        "suite": suite,
        "run_dir": str(run_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary": summary,
    }

# %%
def regenerate_experiment_plots(
    run_dir: Path | str,
    *,
    show: bool = False,
) -> None:
    """
    Regenerate all plots for an existing experiment directory.
    Does NOT retrain models — only reloads logs + metadata.

    Works for:
    - single runs
    - repeated runs (rep_00, rep_01, ...)
    """

    import json
    import matplotlib
    matplotlib.use("Agg")

    run_dir = Path(run_dir).expanduser()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    print(f"Regenerating plots for: {run_dir}")

    def _k_from_variant_slug(slug: str) -> int:
        marker = "__kd"
        if marker not in slug:
            return 0
        suffix = slug.split(marker)[-1]
        try:
            return int(suffix)
        except ValueError:
            return 0

    def _build_model_from_slug(slug: str, input_shape: tuple[int, int, int]) -> keras.Model | None:
        base = canonical_architecture_slug(slug)
        arch_cls = ARCHITECTURE_BY_SLUG.get(base)
        if arch_cls is None:
            return None
        k_deact = _k_from_variant_slug(slug)
        return arch_cls.build(
            input_shape=input_shape,
            kernel_size=KERNEL_SIZE,
            k_deactivated_components=k_deact,
        )

    def _load_weight_pair_logs(logs_dir: Path) -> dict[str, dict[str, Any]]:
        by_model: dict[str, dict[str, Any]] = {}
        if not logs_dir.exists():
            return by_model
        for log_file in sorted(logs_dir.glob("*_weight_pair_logs.npz")):
            suffix = "_weight_pair_logs.npz"
            stem = log_file.name[: -len(suffix)]
            if "_" not in stem:
                continue
            slug, layer_name = stem.rsplit("_", 1)
            data = np.load(log_file, allow_pickle=True)
            model_logs = by_model.setdefault(
                slug, {"epochs": np.asarray([], dtype=np.int32), "layers": {}, "weights_full": {}}
            )
            model_logs["epochs"] = np.asarray(data.get("epochs", []), dtype=np.int32)
            layer_logs: dict[str, np.ndarray] = {}
            for spec in STRUCTURING_ELEMENT_PAIR_SPECS:
                key = spec["key"]
                if key in data:
                    layer_logs[key] = np.asarray(data[key], dtype=np.float32)
            model_logs["layers"][layer_name] = layer_logs
            if "weights_full" in data:
                model_logs["weights_full"][layer_name] = np.asarray(data["weights_full"], dtype=np.float32)
        return by_model

    # --- helper to process one run ---
    def _process_single_run(local_dir: Path):
        print(f"\nProcessing: {local_dir}")

        plots_dir = local_dir / "regenerate_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- load metadata ---
        metadata_path = local_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"Skipping (no metadata): {local_dir}")
            return

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        suite = metadata.get("suite", metadata)  # fallback if structure differs
        metrics_path = local_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}
        if not suite.get("histories"):
            suite["histories"] = metrics.get("histories", {})
        if not suite.get("val_mse"):
            suite["val_mse"] = metrics.get("val_mse", {})
        suite.setdefault("models", {})

        # --- reload dataset (needed for prediction plots) ---
        pack = load_fashion_mnist_four_pixel_pack(seed=DATA_LOAD_SEED)
        _, y_val, _y_test, target_meta = compute_experiment_targets(pack)
        input_shape = (int(pack["H"]), int(pack["W"]), 1)

        # --- rebuild model objects from checkpoints ---
        candidate_slugs: list[str] = []
        for slug in suite.get("histories", {}):
            if slug not in candidate_slugs:
                candidate_slugs.append(slug)
        for slug in suite.get("val_mse", {}):
            if slug not in candidate_slugs:
                candidate_slugs.append(slug)
        for slug in metadata.get("models_trained", []):
            if slug not in candidate_slugs:
                candidate_slugs.append(slug)
        ck_root = local_dir / "checkpoints"
        for slug in candidate_slugs:
            model = _build_model_from_slug(slug, input_shape)
            if model is None:
                continue
            loaded_best = load_best_weights_if_present(model, ck_root, slug)
            if not loaded_best:
                last_path = ck_root / slug / "last.weights.h5"
                if last_path.is_file():
                    model.load_weights(str(last_path))
                    loaded_best = True
            if loaded_best:
                suite["models"][slug] = model

        # --- load weight-pair logs in the same nested format used by plotting ---
        suite["weight_pair_logs"] = _load_weight_pair_logs(local_dir / "weight_pair_logs")

        # --- regenerate comparison plots ---
        run_comparison_plots(
            suite,
            pack,
            y_val,
            plots_dir,
            target_label=target_meta["target_label"],
            show=show,
            n_pred_samples=min(4, len(y_val)),
        )

        print(f"Done: {local_dir}")

    # --- detect repeated runs ---
    rep_dirs = sorted([p for p in run_dir.glob("rep_*") if p.is_dir()])

    if rep_dirs:
        print(f"Detected repeated experiment with {len(rep_dirs)} runs")
        for rep_dir in rep_dirs:
            _process_single_run(rep_dir)
    else:
        _process_single_run(run_dir)

    print("\nAll plots regenerated.")


# %%
# def main(
#     out_dir: Path | str | None = None,
#     *,
#     notebook: bool = False,
#     dated_subdir: bool = True,
# ) -> dict:
#     return run_experiment(
#         out_dir=out_dir, notebook=notebook, dated_subdir=dated_subdir
#     )

# if __name__ == "__main__":
#     main(notebook=False, dated_subdir=True)

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
run_experiment(
        out_dir='/content/outputs', notebook=False, dated_subdir=True
    )

# %%
run_k_deactivated_initialization_experiment(
    out_dir='/content/outputs', notebook=False, dated_subdir=True
)

# %%
# !cp -r '/content/outputs/exp_four_pixel_average_k_deactivated_grid_2026-04-21_221420/' '/content/drive/MyDrive/experiments-dima/'

# %%
# !cp -r '/content/drive/MyDrive/experiments-dima/outputs/exp_four_pixel_average_k_deactivated_grid_2026-04-21_221420' '/content/outputs/exp_four_pixel_average_k_deactivated_grid_2026-04-21_221420'

# %%
regenerate_experiment_plots("/content/outputs/exp_four_pixel_average_k_deactivated_grid_2026-04-21_221420")
