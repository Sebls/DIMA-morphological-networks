"""Register and build initializers from config ``method`` / ``strategy`` keys."""

from __future__ import annotations

from typing import Any, Callable

import keras

from morpho_net.initialization.minimal_e_patches import MinimalEPatchesInitializer
from morpho_net.initialization.random_e_patches import RandomEPatchesInitializer
from morpho_net.initialization.uniform import UniformMorphoInitializer

InitializerBuilder = Callable[[dict[str, Any]], keras.initializers.Initializer]

INITIALIZER_REGISTRY: dict[str, InitializerBuilder] = {
    "uniform": UniformMorphoInitializer.from_config,
    "random_e_patches": RandomEPatchesInitializer.from_config,
    "minimal_e_patches": MinimalEPatchesInitializer.from_config,
}


def register_initializer(name: str, builder: InitializerBuilder) -> None:
    """Register or replace an initializer name (for plugins / future methods)."""
    INITIALIZER_REGISTRY[name] = builder


def list_initializers() -> tuple[str, ...]:
    return tuple(sorted(INITIALIZER_REGISTRY))


def build_kernel_initializer(merged_cfg: dict[str, Any]) -> keras.initializers.Initializer:
    """Instantiate a Keras initializer from merged YAML options."""
    method = merged_cfg.get("method") or merged_cfg.get("strategy") or "uniform"
    if method not in INITIALIZER_REGISTRY:
        known = ", ".join(sorted(INITIALIZER_REGISTRY))
        raise ValueError(f"Unknown initialization method: {method!r}. Known: {known}")
    return INITIALIZER_REGISTRY[method](merged_cfg)
