"""Kernel and structuring element visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _paginated_save_path(
    base: Path, n_pages: int, page_index: int
) -> Path:
    """Single path when one page; ``name_p01.png`` style when multiple pages."""
    base = Path(base)
    if n_pages <= 1:
        return base
    return base.parent / f"{base.stem}_p{page_index + 1:02d}{base.suffix}"


def _page_ranges(n_items: int, filters_per_page: int | None) -> list[tuple[int, int]]:
    if filters_per_page is None or filters_per_page <= 0:
        return [(0, n_items)]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n_items:
        end = min(start + filters_per_page, n_items)
        ranges.append((start, end))
        start = end
    return ranges


def plot_structuring_elements(
    weights: np.ndarray,
    kernel_shape: tuple[int, int] = (3, 3),
    n_show: int | None = None,
    cols: int = 10,
    filters_per_page: int | None = None,
    title: str = "Learned Structuring Elements",
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[Path]:
    """Plot learned structuring elements from erosion layer weights.

    Args:
        weights: From get_weights()[0], shape (1,1,1, patch_size, n_filters).
        kernel_shape: (H, W) of each kernel.
        n_show: Max number to display; None = all.
        cols: Number of columns in each page's grid.
        filters_per_page: If set (e.g. 25), split into multiple figures/files with at most
            this many kernels per page. ``None`` = one figure for all ``n_show`` kernels.
        title: Figure title (page suffix ``(page i / n)`` added when paginated).
        save_path: Optional path to save figure(s).
        show: Whether to display figures (each page is its own figure when paginated).

    Returns:
        List of paths written when ``save_path`` is set; empty list otherwise.
    """
    flat = weights[0, 0, 0, :, :]  # (patch_size, n_filters)
    kernels = flat.T.reshape(-1, *kernel_shape)
    n_total = kernels.shape[0]
    n_show = n_show or n_total
    n_show = min(n_show, n_total)

    pages = _page_ranges(n_show, filters_per_page)
    saved: list[Path] = []
    vmin, vmax = -1.1, 1.1

    for page_idx, (start, end) in enumerate(pages):
        n_this = end - start
        rows = (n_this + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        n_cells = rows * cols

        for slot in range(n_this):
            global_i = start + slot
            ax = axes_flat[slot]
            k = kernels[global_i]
            ax.imshow(k, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            for r in range(kernel_shape[0]):
                for c in range(kernel_shape[1]):
                    val = k[r, c]
                    color = "blue" if val > 0 else "red"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
            ax.set_title(f"Kernel {global_i}", fontsize=8)
            ax.axis("off")

        for j in range(n_this, n_cells):
            axes_flat[j].axis("off")

        page_title = title
        if len(pages) > 1:
            page_title = f"{title} — page {page_idx + 1} / {len(pages)}"
        plt.suptitle(page_title, fontsize=14, y=1.00)
        plt.tight_layout()
        if save_path:
            out = _paginated_save_path(Path(save_path), len(pages), page_idx)
            plt.savefig(out, bbox_inches="tight", dpi=150)
            saved.append(out)
        if show:
            plt.show()
        else:
            plt.close()

    return saved


def plot_pareto_elements(
    filters: np.ndarray,
    kernel_shape: tuple[int, int] = (3, 3),
    cols: int = 10,
    filters_per_page: int | None = None,
    title: str = "Minimal Structuring Elements",
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[Path]:
    """Plot Pareto-minimal structuring elements.

    Args:
        filters: (patch_size, n_filters) array.
        kernel_shape: (H, W) of each kernel.
        cols: Number of columns in each page's grid.
        filters_per_page: If set, split into multiple figures/files with at most this many
            filters per page. ``None`` = one figure for all filters.
        title: Figure title (page suffix added when paginated).
        save_path: Optional path to save figure(s).

    Returns:
        List of paths written when ``save_path`` is set; empty list otherwise.
    """
    n_filters = filters.shape[1]
    kernels = filters.T.reshape(-1, *kernel_shape)

    pages = _page_ranges(n_filters, filters_per_page)
    saved: list[Path] = []
    vmin, vmax = np.min(filters), np.max(filters)

    for page_idx, (start, end) in enumerate(pages):
        n_this = end - start
        rows = (n_this + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        n_cells = rows * cols

        for slot in range(n_this):
            i = start + slot
            ax = axes_flat[slot]
            k = kernels[i]
            ax.imshow(k, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            for r in range(kernel_shape[0]):
                for c in range(kernel_shape[1]):
                    val = k[r, c]
                    color = "blue" if val > 0 else "red"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
            ax.axis("off")

        for j in range(n_this, n_cells):
            axes_flat[j].axis("off")

        page_title = title
        if len(pages) > 1:
            page_title = f"{title} — page {page_idx + 1} / {len(pages)}"
        plt.suptitle(page_title, fontsize=14, y=1.00)
        plt.tight_layout()
        if save_path:
            out = _paginated_save_path(Path(save_path), len(pages), page_idx)
            plt.savefig(out, bbox_inches="tight", dpi=150)
            saved.append(out)
        if show:
            plt.show()
        else:
            plt.close()

    return saved
