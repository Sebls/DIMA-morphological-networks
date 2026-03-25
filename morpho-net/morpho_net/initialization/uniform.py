"""Uniform random initialization (default RandomUniform-style range)."""

from __future__ import annotations

from typing import Any

import keras

from morpho_net.initialization.base import MorphoInitializer, _scalar_uniform


@keras.saving.register_keras_serializable(package="MorphoNet")
class UniformMorphoInitializer(MorphoInitializer):
    """Independent uniform samples in ``[minval, maxval]`` for every weight."""

    def __init__(self, minval: float = -0.45, maxval: float = -0.15, seed: int | None = None):
        self.minval = float(minval)
        self.maxval = float(maxval)
        self.seed = seed

    def __call__(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        if len(shape) == 1:
            return _scalar_uniform(shape, dtype, self.minval, self.maxval, self.seed)
        return keras.random.uniform(
            shape, minval=self.minval, maxval=self.maxval, seed=self.seed, dtype=dtype
        )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "minval": self.minval,
            "maxval": self.maxval,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> UniformMorphoInitializer:
        return cls(
            minval=float(cfg.get("minval", -0.45)),
            maxval=float(cfg.get("maxval", -0.15)),
            seed=cfg.get("seed"),
        )
