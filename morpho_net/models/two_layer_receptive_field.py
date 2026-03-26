"""Two-layer SupErosions with receptive field specialization."""

from __future__ import annotations

from typing import Any

import keras
from keras import layers, Model, Input

from morpho_net.initialization import build_kernel_initializer, merge_init_block
from morpho_net.layers.dilation import MorphologicalDilation
from morpho_net.layers.sup_erosions import SupErosionsBlock2Inputs


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
    """Build two-layer model with zone-specialized receptive fields.

    Block1: weights at block1_inactive_indices set to inactive_value.
    Block2: weights at block2_inactive_indices set to inactive_value.
    """
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    block1_inactive_indices = block1_inactive_indices or []
    block2_inactive_indices = block2_inactive_indices or []

    x = Input(shape=input_shape, name="input")
    x_pad = layers.ZeroPadding2D(padding=padding)(x)

    # Block 1
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

    # Block 2
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

    # Apply receptive field masks after build
    _apply_receptive_field_masks(
        model,
        block1_inactive_indices,
        block2_inactive_indices,
        inactive_value,
    )

    return model


def _apply_receptive_field_masks(
    model: Model,
    block1_inactive: list[int],
    block2_inactive: list[int],
    inactive_value: float,
) -> None:
    """Set specified weight positions to inactive_value for zone specialization."""
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


def build_from_config(model_cfg: dict[str, Any], init_cfg: dict[str, Any]) -> Model:
    """Instantiate from ``model`` / ``initialization`` config dicts (YAML sections)."""
    m1 = merge_init_block(init_cfg, "block1")
    m2 = merge_init_block(init_cfg, "block2")
    m3 = merge_init_block(init_cfg, "block3")
    return build_two_layer_receptive_field(
        n_erosions_block1=model_cfg.get("n_erosions_block1", 500),
        n_erosions_block2=model_cfg.get("n_erosions_block2", 500),
        n_erosions_block3=model_cfg.get("n_erosions_block3", 700),
        kernel_size=tuple(model_cfg.get("kernel_size", [3, 3])),
        block1_inactive_indices=model_cfg.get("block1_inactive_indices", [4, 5, 6, 7, 8]),
        block2_inactive_indices=model_cfg.get("block2_inactive_indices", [0, 1, 2, 3]),
        inactive_value=model_cfg.get("inactive_value", -10.0),
        init_block1=(
            float(m1.get("minval", -0.35)),
            float(m1.get("maxval", 0.35)),
        ),
        init_block2=(
            float(m2.get("minval", -0.35)),
            float(m2.get("maxval", 0.35)),
        ),
        init_block3=(
            float(m3.get("minval", -0.35)),
            float(m3.get("maxval", 0.35)),
        ),
        seed=init_cfg.get("seed"),
        kernel_initializer_block1=build_kernel_initializer(m1),
        kernel_initializer_block2=build_kernel_initializer(m2),
        weight_initializer_block3=build_kernel_initializer(m3),
    )
