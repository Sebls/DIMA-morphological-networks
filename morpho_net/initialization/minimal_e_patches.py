"""Initialize dilation filters from a Pareto-minimal subset of random candidates."""

from __future__ import annotations

from typing import Any

import keras
import numpy as np

from morpho_net.analysis.pareto import is_pareto_efficient
from morpho_net.initialization.base import MorphoInitializer, _scalar_uniform


@keras.saving.register_keras_serializable(package="MorphoNet")
class MinimalEPatchesInitializer(MorphoInitializer):
    """Draw many random patch vectors, keep Pareto-minimal ones, subsample to ``n_filters``.

    Coordinates are treated as costs to minimize (same convention as
    :mod:`morpho_net.analysis.pareto`). For non-dilation shapes (1D pairing
    weights), falls back to uniform scalar init.
    """

    def __init__(
        self,
        minval: float = -0.35,
        maxval: float = 0.35,
        oversample_factor: float = 8.0,
        seed: int | None = None,
    ):
        self.minval = float(minval)
        self.maxval = float(maxval)
        self.oversample_factor = float(oversample_factor)
        self.seed = seed

    def __call__(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        if len(shape) != 5:
            return _scalar_uniform(shape, dtype, self.minval, self.maxval, self.seed)

        _b0, _b1, _b2, patch_size, n_filters = shape
        rng = np.random.default_rng(self.seed)
        n_candidates = max(int(n_filters * self.oversample_factor), n_filters + 1)
        candidates = rng.uniform(self.minval, self.maxval, size=(n_candidates, patch_size)).astype(
            np.float32
        )
        mask = is_pareto_efficient(candidates.copy(), return_mask=True)
        pareto = candidates[mask]
        if pareto.shape[0] >= n_filters:
            choice_idx = rng.choice(pareto.shape[0], size=n_filters, replace=False)
            selected = pareto[choice_idx]
        else:
            extra = rng.uniform(
                self.minval, self.maxval, size=(n_filters - pareto.shape[0], patch_size)
            ).astype(np.float32)
            selected = np.vstack([pareto, extra]) if pareto.shape[0] else extra
        w = selected.T.reshape(1, 1, 1, patch_size, n_filters)
        return keras.ops.cast(w, dtype=dtype or keras.backend.floatx())

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "minval": self.minval,
            "maxval": self.maxval,
            "oversample_factor": self.oversample_factor,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MinimalEPatchesInitializer:
        return cls(
            minval=float(cfg.get("minval", -0.35)),
            maxval=float(cfg.get("maxval", 0.35)),
            oversample_factor=float(cfg.get("oversample_factor", 8.0)),
            seed=cfg.get("seed"),
        )
