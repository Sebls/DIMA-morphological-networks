"""Random structuring-element patches (per-weight randomness with optional distribution)."""

from __future__ import annotations

from typing import Any, Literal

import keras

from morpho_net.initialization.base import MorphoInitializer, _scalar_uniform

Distribution = Literal["uniform", "normal"]


@keras.saving.register_keras_serializable(package="MorphoNet")
class RandomEPatchesInitializer(MorphoInitializer):
    """Random values for each patch weight; supports uniform or normal.

    For dilation-shaped tensors this matches independent draws per entry. For 1D
    pairing weights, falls back to the same distribution with scalar shape.
    """

    def __init__(
        self,
        distribution: Distribution = "uniform",
        minval: float = -0.35,
        maxval: float = 0.35,
        mean: float = 0.0,
        stddev: float = 0.2,
        seed: int | None = None,
    ):
        self.distribution = distribution
        self.minval = float(minval)
        self.maxval = float(maxval)
        self.mean = float(mean)
        self.stddev = float(stddev)
        self.seed = seed

    def __call__(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        if len(shape) == 1:
            if self.distribution == "uniform":
                return _scalar_uniform(shape, dtype, self.minval, self.maxval, self.seed)
            return keras.random.normal(shape, mean=self.mean, stddev=self.stddev, seed=self.seed, dtype=dtype)

        if self.distribution == "uniform":
            return keras.random.uniform(
                shape, minval=self.minval, maxval=self.maxval, seed=self.seed, dtype=dtype
            )
        return keras.random.normal(shape, mean=self.mean, stddev=self.stddev, seed=self.seed, dtype=dtype)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "distribution": self.distribution,
            "minval": self.minval,
            "maxval": self.maxval,
            "mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RandomEPatchesInitializer:
        return cls(
            distribution=cfg.get("distribution", "uniform"),
            minval=float(cfg.get("minval", -0.35)),
            maxval=float(cfg.get("maxval", 0.35)),
            mean=float(cfg.get("mean", 0.0)),
            stddev=float(cfg.get("stddev", 0.2)),
            seed=cfg.get("seed"),
        )
