"""Two-layer Supremum of Erosions model."""

from __future__ import annotations

from typing import Any

import keras
from keras import layers, Model, Input

from morpho_net.initialization import build_kernel_initializer, merge_init_block
from morpho_net.layers.dilation import MorphologicalDilation
from morpho_net.layers.sup_erosions import SupErosionsBlock2Inputs


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
    """Build two-layer SupErosions model.

    Layer 1: Two parallel SupErosions blocks on same input.
    Layer 2: SupErosions block combining both outputs.
    """
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)

    x = Input(shape=input_shape, name="input")
    x_pad = layers.ZeroPadding2D(padding=padding)(x)

    # Block 1
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

    # Block 2
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

    # Layer 2: combine
    out = SupErosionsBlock2Inputs(
        n_erosions=n_erosions_block3,
        minval=init_block3[0],
        maxval=init_block3[1],
        seed=seed,
        weight_initializer=weight_initializer_block3,
        name="SupErosions_3",
    )([sup1, sup2])

    return Model(x, out, name=name)


def build_from_config(model_cfg: dict[str, Any], init_cfg: dict[str, Any]) -> Model:
    """Instantiate from ``model`` / ``initialization`` config dicts (YAML sections)."""
    m1 = merge_init_block(init_cfg, "block1")
    m2 = merge_init_block(init_cfg, "block2")
    m3 = merge_init_block(init_cfg, "block3")
    return build_two_layer_sup_erosions(
        n_erosions_block1=model_cfg.get("n_erosions_block1", 50),
        n_erosions_block2=model_cfg.get("n_erosions_block2", 50),
        n_erosions_block3=model_cfg.get("n_erosions_block3", 100),
        kernel_size=tuple(model_cfg.get("kernel_size", [3, 3])),
        init_block1=(
            float(m1.get("minval", -0.45)),
            float(m1.get("maxval", -0.15)),
        ),
        init_block2=(
            float(m2.get("minval", -0.45)),
            float(m2.get("maxval", -0.15)),
        ),
        init_block3=(
            float(m3.get("minval", -0.45)),
            float(m3.get("maxval", -0.15)),
        ),
        seed=init_cfg.get("seed"),
        kernel_initializer_block1=build_kernel_initializer(m1),
        kernel_initializer_block2=build_kernel_initializer(m2),
        weight_initializer_block3=build_kernel_initializer(m3),
    )
