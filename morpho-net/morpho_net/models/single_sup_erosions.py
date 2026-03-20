"""Single-layer Supremum of Erosions model."""

from __future__ import annotations

import keras
from keras import layers, Model, Input

from morpho_net.layers.dilation import MorphologicalDilation


def build_single_sup_erosions(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    n_erosions: int = 200,
    kernel_size: tuple[int, int] = (3, 3),
    minval: float = -0.45,
    maxval: float = -0.15,
    seed: int | None = None,
    name: str = "single_sup_erosions",
) -> Model:
    """Build single SupErosions model.

    Architecture: Input -> Pad -> Erosion block -> max over filters -> Output
    Erosion implemented as -Dilation(-x).
    """
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)

    x = Input(shape=input_shape, name="input")
    x_pad = layers.ZeroPadding2D(padding=padding)(x)
    erosion_out = -MorphologicalDilation(
        filters=n_erosions,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding="VALID",
        minval=minval,
        maxval=maxval,
        seed=seed,
        name="Erosion",
    )(-x_pad)
    out = keras.ops.max(erosion_out, axis=-1, keepdims=True)

    return Model(x, out, name=name)
