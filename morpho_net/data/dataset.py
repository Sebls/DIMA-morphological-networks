"""Fashion-MNIST dataset loading with optional noise."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from keras.datasets import fashion_mnist


class DatasetSplit(NamedTuple):
    """Data split with images and optional labels."""

    images: np.ndarray
    labels: np.ndarray | None = None


def load_fashion_mnist_noisy(
    size: int = 100,
    use_noise: bool = True,
    sigma: float = 40.0,
    seed: int | None = 42,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Load Fashion-MNIST with optional Gaussian noise.

    Returns train, validation, and test splits. Each split has `size` samples.
    Validation uses x_test[0:size], test uses x_test[size:2*size].

    Args:
        size: Number of samples per split.
        use_noise: Whether to add Gaussian noise.
        sigma: Standard deviation of noise.
        seed: Seed for noise only (local ``numpy.random.Generator``). Does not
            alter the global NumPy RNG, so experiment-level seeds stay intact.

    Returns:
        (train_split, val_split, test_split) with normalized images [0, 1].
    """
    rng = np.random.default_rng(seed)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    X_train = x_train[0:size].astype(np.float32)
    X_val = x_test[0:size].astype(np.float32)
    X_test = x_test[size : 2 * size].astype(np.float32)

    def add_noise(x: np.ndarray) -> np.ndarray:
        if use_noise:
            noisy = x + rng.normal(0, sigma, x.shape)
        else:
            noisy = x.copy()
        return np.clip(noisy, 0, 255).astype(np.float32)

    x_train_noisy = add_noise(X_train)
    x_val_noisy = add_noise(X_val)
    x_test_noisy = add_noise(X_test)

    # Normalize to [0, 1] and add channel dimension (N, H, W) -> (N, H, W, 1)
    def add_channel(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return np.expand_dims(x / 255.0, axis=(0, -1))
        return np.expand_dims(x / 255.0, axis=-1)

    train_split = DatasetSplit(
        images=add_channel(x_train_noisy),
        labels=y_train[0:size],
    )
    val_split = DatasetSplit(
        images=add_channel(x_val_noisy),
        labels=y_test[0:size],
    )
    test_split = DatasetSplit(
        images=add_channel(x_test_noisy),
        labels=y_test[size : 2 * size],
    )

    return train_split, val_split, test_split
