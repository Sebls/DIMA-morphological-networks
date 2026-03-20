"""Morphological dilation layer.

Dilation: (f ⊕ b)(x) = max_{y ∈ b} f(x + y)
Implemented via patch extraction: for each patch P, output = max(P + w) over patch positions.
"""

from __future__ import annotations

import keras
from keras import layers


class MorphologicalDilation(layers.Layer):
    """Learnable morphological dilation via flat structuring elements.

    Weight shape: (1, 1, 1, kernel_size^2, filters)
    Forward: extract patches, add weights, take max over patch dimension.
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
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def build(self, input_shape: tuple) -> None:
        k_h, k_w = self.kernel_size
        patch_size = k_h * k_w
        self.w = self.add_weight(
            shape=(1, 1, 1, patch_size, self.filters),
            initializer=keras.initializers.RandomUniform(
                minval=self.minval,
                maxval=self.maxval,
                seed=self.seed,
            ),
            trainable=True,
            name="dilation_weights",
        )
        super().build(input_shape)

    def call(self, inputs):
        patches = keras.ops.image.extract_patches(
            inputs,
            self.kernel_size,
            strides=self.stride,
            dilation_rate=1,
            padding=self.padding,
        )
        patches = keras.ops.expand_dims(patches, axis=-1)
        return keras.ops.max(patches + self.w, axis=3)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "minval": self.minval,
            "maxval": self.maxval,
            "seed": self.seed,
        })
        return config
