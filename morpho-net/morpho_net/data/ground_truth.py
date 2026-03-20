"""Ground truth generation from convolution kernels."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf


class GroundTruthConfig(NamedTuple):
    """Configuration for ground truth convolution."""

    kernel: list[list[float]]


def create_ground_truth_model(kernel: list[list[float]]) -> tf.keras.Model:
    """Create a Conv2D model that applies the given convolution kernel.

    Args:
        kernel: 3x3 kernel as list of lists.

    Returns:
        Keras model with single Conv2D layer.
    """
    matrix = np.array(kernel, dtype=np.float32)
    matrix = matrix.reshape((3, 3, 1, 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", use_bias=False),
    ])
    model.layers[0].set_weights([matrix])
    return model


def generate_ground_truth(
    model: tf.keras.Model,
    train_images: np.ndarray,
    val_images: np.ndarray,
    test_images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ground truth targets from model predictions.

    Args:
        model: Ground truth model.
        train_images: Train images (H, W) or (N, H, W).
        val_images: Validation images.
        test_images: Test images.

    Returns:
        (y_train, y_val, y_test) as (N, H, W) arrays.
    """
    def expand_and_predict(images: np.ndarray) -> np.ndarray:
        if images.ndim == 2:
            images = np.expand_dims(images, axis=(0, -1))
        elif images.ndim == 3:
            images = np.expand_dims(images, axis=-1)
        pred = model.predict(images, verbose=0)
        return pred[:, :, :, 0]

    y_train = expand_and_predict(train_images)
    y_val = expand_and_predict(val_images)
    y_test = expand_and_predict(test_images)

    return y_train, y_val, y_test
