"""Architecture registry: map config ``architecture`` names to model factories."""

from __future__ import annotations

from typing import Any, Callable

import keras

from morpho_net.models.single_sup_erosions import build_from_config as _build_single_sup_erosions
from morpho_net.models.two_layer_sup_erosions import build_from_config as _build_two_layer_sup_erosions
from morpho_net.models.two_layer_receptive_field import build_from_config as _build_two_layer_receptive_field

ModelBuilder = Callable[[dict[str, Any], dict[str, Any]], keras.Model]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "single_sup_erosions": _build_single_sup_erosions,
    "two_layer_sup_erosions": _build_two_layer_sup_erosions,
    "two_layer_receptive_field": _build_two_layer_receptive_field,
}


def register_architecture(name: str, builder: ModelBuilder) -> None:
    """Register or override an architecture name (e.g. from an extension module)."""
    MODEL_REGISTRY[name] = builder


def build_model(
    architecture: str,
    model_cfg: dict[str, Any],
    init_cfg: dict[str, Any],
) -> keras.Model:
    """Build a model from YAML ``model`` and ``initialization`` sections."""
    builder = MODEL_REGISTRY.get(architecture)
    if builder is None:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown architecture: {architecture!r}. Known: {known}")
    return builder(model_cfg, init_cfg)


def list_architectures() -> tuple[str, ...]:
    """Registered architecture names."""
    return tuple(sorted(MODEL_REGISTRY))
