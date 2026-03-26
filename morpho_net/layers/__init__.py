"""Core morphological operators."""

from morpho_net.layers.dilation import MorphologicalDilation
from morpho_net.layers.erosion import MorphologicalErosion
from morpho_net.layers.sup_erosions import SupErosionsBlock, SupErosionsBlock2Inputs

__all__ = [
    "MorphologicalDilation",
    "MorphologicalErosion",
    "SupErosionsBlock",
    "SupErosionsBlock2Inputs",
]
