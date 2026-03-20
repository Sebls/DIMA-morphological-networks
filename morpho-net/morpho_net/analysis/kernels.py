"""Kernel and structuring element visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_structuring_elements(
    weights: np.ndarray,
    kernel_shape: tuple[int, int] = (3, 3),
    n_show: int | None = None,
    cols: int = 10,
    title: str = "Learned Structuring Elements",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot learned structuring elements from erosion layer weights.

    Args:
        weights: From get_weights()[0], shape (1,1,1, patch_size, n_filters).
        kernel_shape: (H, W) of each kernel.
        n_show: Max number to display; None = all.
        cols: Number of columns in grid.
        title: Figure title.
        save_path: Optional path to save figure.
    """
    flat = weights[0, 0, 0, :, :]  # (patch_size, n_filters)
    kernels = flat.T.reshape(-1, *kernel_shape)
    n_total = kernels.shape[0]
    n_show = n_show or n_total
    n_show = min(n_show, n_total)

    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    vmin, vmax = -1.1, 1.1
    for i in range(n_show):
        ax = axes_flat[i]
        k = kernels[i]
        im = ax.imshow(k, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        for r in range(kernel_shape[0]):
            for c in range(kernel_shape[1]):
                val = k[r, c]
                color = "blue" if val > 0 else "red"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.set_title(f"Kernel {i}", fontsize=8)
        ax.axis("off")

    for j in range(n_show, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_pareto_elements(
    filters: np.ndarray,
    kernel_shape: tuple[int, int] = (3, 3),
    title: str = "Minimal Structuring Elements",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot Pareto-minimal structuring elements.

    Args:
        filters: (patch_size, n_filters) array.
        kernel_shape: (H, W) of each kernel.
    """
    n_filters = filters.shape[1]
    kernels = filters.T.reshape(-1, *kernel_shape)

    cols = 10
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    vmin, vmax = np.min(filters), np.max(filters)
    for i in range(n_filters):
        ax = axes_flat[i]
        k = kernels[i]
        ax.imshow(k, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        for r in range(kernel_shape[0]):
            for c in range(kernel_shape[1]):
                val = k[r, c]
                color = "blue" if val > 0 else "red"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.axis("off")

    for j in range(n_filters, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
