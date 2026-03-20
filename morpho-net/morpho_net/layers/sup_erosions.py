"""Supremum of erosions layers.

SupErosions: sup over multiple erosions with different structuring elements.
"""

from __future__ import annotations

import keras
from keras import layers
from keras import ops
import tensorflow as tf

from morpho_net.layers.dilation import MorphologicalDilation


class SupErosionsBlock(layers.Layer):
    """Single-input Supremum of Erosions block.

    Applies n_erosions parallel erosions and takes the maximum.
    Input: (N, H, W, C) -> Output: (N, H, W, 1)
    """

    def __init__(
        self,
        n_erosions: int,
        kernel_size: tuple[int, int] = (3, 3),
        minval: float = -0.45,
        maxval: float = -0.15,
        padding: str = "VALID",
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_erosions = n_erosions
        self.kernel_size = kernel_size
        self.minval = minval
        self.maxval = maxval
        self.padding = padding
        self.seed = seed
        self.erosion = MorphologicalDilation(  # erosion = -dilation(-x)
            filters=n_erosions,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            minval=minval,
            maxval=maxval,
            seed=seed,
        )

    def build(self, input_shape: tuple) -> None:
        self.erosion.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        # erosion(x) = -dilation(-x)
        out = -self.erosion(-inputs)
        return ops.max(out, axis=-1, keepdims=True)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "n_erosions": self.n_erosions,
            "kernel_size": self.kernel_size,
            "minval": self.minval,
            "maxval": self.maxval,
            "padding": self.padding,
            "seed": self.seed,
        })
        return config


class SupErosionsBlock2Inputs(layers.Layer):
    """Two-input Supremum of Erosions block (layer 2).

    Takes two (N, H, W, 1) tensors, applies n_erosions paired erosions
    (min over inputs per erosion), then supremum over erosions.
    Output: (N, H, W, 1)
    """

    def __init__(
        self,
        n_erosions: int,
        minval: float = -0.45,
        maxval: float = -0.15,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_erosions = n_erosions
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def build(self, input_shape) -> None:
        # input_shape can be list of two shapes for two inputs
        self.w1 = self.add_weight(
            shape=(self.n_erosions,),
            initializer=keras.initializers.RandomUniform(
                minval=self.minval,
                maxval=self.maxval,
                seed=self.seed,
            ),
            trainable=True,
            name="w1",
        )
        self.w2 = self.add_weight(
            shape=(self.n_erosions,),
            initializer=keras.initializers.RandomUniform(
                minval=self.minval,
                maxval=self.maxval,
                seed=self.seed,
            ),
            trainable=True,
            name="w2",
        )
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            input1, input2 = inputs
        else:
            raise ValueError("SupErosionsBlock2Inputs expects [input1, input2]")

        # input1, input2: (N, H, W, 1) -> broadcast to (N, H, W, n_erosions)
        z1 = input1 - self.w1
        z2 = input2 - self.w2
        t = tf.minimum(z1, z2)  # Erosions
        return tf.reduce_max(t, axis=-1, keepdims=True)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "n_erosions": self.n_erosions,
            "minval": self.minval,
            "maxval": self.maxval,
            "seed": self.seed,
        })
        return config
