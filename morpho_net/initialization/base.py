"""Base class for morphological weight initializers.

Subclasses implement ``__call__(shape, dtype=None)`` for Keras ``add_weight``.
Dilation kernels use shape ``(1, 1, 1, patch_size, n_filters)``; scalar pairing
weights use shape ``(n_erosions,)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import keras


class MorphoInitializer(keras.initializers.Initializer, ABC):
    """Abstract initializer for morphological layer weights."""

    @abstractmethod
    def __call__(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Return a tensor of the given shape and dtype."""

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict[str, Any]) -> MorphoInitializer:
        """Build from a merged YAML-style dict (single block or global defaults)."""


def _scalar_uniform(
    shape: tuple[int, ...],
    dtype: Any,
    minval: float,
    maxval: float,
    seed: int | None,
) -> Any:
    """Fallback for 1D weights (e.g. SupErosionsBlock2Inputs)."""
    return keras.random.uniform(shape, minval=minval, maxval=maxval, seed=seed, dtype=dtype)
