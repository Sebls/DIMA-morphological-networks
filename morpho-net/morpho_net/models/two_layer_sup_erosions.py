"""Two-layer Supremum of Erosions model."""

from __future__ import annotations

import keras
from keras import layers, Model, Input

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
        name="Erosions2",
    )(-x_pad)
    sup2 = keras.ops.max(out2, axis=-1, keepdims=True)

    # Layer 2: combine
    out = SupErosionsBlock2Inputs(
        n_erosions=n_erosions_block3,
        minval=init_block3[0],
        maxval=init_block3[1],
        seed=seed,
        name="SupErosions_3",
    )([sup1, sup2])

    return Model(x, out, name=name)
