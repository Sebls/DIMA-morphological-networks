"""Network architectures."""

from morpho_net.models.single_sup_erosions import build_single_sup_erosions
from morpho_net.models.two_layer_sup_erosions import build_two_layer_sup_erosions
from morpho_net.models.two_layer_receptive_field import build_two_layer_receptive_field

__all__ = [
    "build_single_sup_erosions",
    "build_two_layer_sup_erosions",
    "build_two_layer_receptive_field",
]
