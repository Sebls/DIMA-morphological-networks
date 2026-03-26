"""Weight initialization strategies for morphological layers (config-driven, extensible)."""

from morpho_net.utils.merge import merge_init_block
from morpho_net.initialization.minimal_e_patches import MinimalEPatchesInitializer
from morpho_net.initialization.random_e_patches import RandomEPatchesInitializer
from morpho_net.initialization.registry import (
    INITIALIZER_REGISTRY,
    build_kernel_initializer,
    list_initializers,
    register_initializer,
)
from morpho_net.initialization.uniform import UniformMorphoInitializer

__all__ = [
    "INITIALIZER_REGISTRY",
    "MinimalEPatchesInitializer",
    "RandomEPatchesInitializer",
    "UniformMorphoInitializer",
    "build_kernel_initializer",
    "list_initializers",
    "merge_init_block",
    "register_initializer",
]
