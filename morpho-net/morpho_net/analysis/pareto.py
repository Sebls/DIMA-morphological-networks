"""Pareto frontier computation for minimal structuring elements."""

from __future__ import annotations

import numpy as np

# Only the minimal elements contribute to the representation- we can remove the rest
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python#40239615

def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
    """Find Pareto-efficient points (minimal elements).

    A point is Pareto-efficient if no other point dominates it
    (i.e., has all coordinates <= and at least one <).

    Args:
        costs: (n_points, n_costs) array
        return_mask: If True, return boolean mask; else return indices.

    Returns:
        Boolean mask of shape (n_points,) or integer indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0 # Next index in the is_efficient array to search for

    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask] # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    return is_efficient


def extract_pareto_filters(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract Pareto-efficient filters from erosion layer weights.

    Args:
        weights: From get_weights()[0], shape (1,1,1, patch_size, n_filters).

    Returns:
        (pareto_filters, pareto_mask) where pareto_filters has shape (patch_size, n_pareto).
    """
    # weights: (1,1,1, patch_size, n_filters) -> we want (n_filters, patch_size)
    flat = weights[0, 0, 0, :, :]  # (patch_size, n_filters)
    filters_t = flat.T  # (n_filters, patch_size)
    mask = is_pareto_efficient(filters_t.copy(), return_mask=True)
    return flat[:, mask], mask
