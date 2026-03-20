"""Morphological erosion layer.

Erosion: (f ⊖ b)(x) = min_{y ∈ b} f(x + y)
Duality: erosion(f) = -dilation(-f)
"""

from __future__ import annotations

import keras
from keras import layers

from morpho_net.layers.dilation import MorphologicalDilation


class MorphologicalErosion(layers.Layer):
    """Learnable morphological erosion via duality with dilation.

    erosion(x) = -dilation(-x)
    """

    def __init__(
        self,
        filters: int = 32,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: str = "VALID",
        minval: float = -0.45,
        maxval: float = -0.15,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dilation = MorphologicalDilation(
            filters=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            minval=minval,
            maxval=maxval,
            seed=seed,
        )

    def build(self, input_shape: tuple) -> None:
        self.dilation.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return -self.dilation(-inputs)

    def get_config(self) -> dict:
        return self.dilation.get_config()
