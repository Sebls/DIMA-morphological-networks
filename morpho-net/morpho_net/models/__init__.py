"""Network architectures."""

from morpho_net.models.registry import (
    MODEL_REGISTRY,
    build_model,
    list_architectures,
    register_architecture,
)
from morpho_net.models.single_sup_erosions import build_single_sup_erosions
from morpho_net.models.two_layer_sup_erosions import build_two_layer_sup_erosions
from morpho_net.models.two_layer_receptive_field import build_two_layer_receptive_field

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "build_single_sup_erosions",
    "build_two_layer_receptive_field",
    "build_two_layer_sup_erosions",
    "list_architectures",
    "register_architecture",
]
