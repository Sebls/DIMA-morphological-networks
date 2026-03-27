"""Plots and saved plot data from weight snapshots across training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from morpho_net.analysis.pareto import extract_pareto_filters


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=True)


def _split_weights_npz_keys(data: np.lib.npyio.NpzFile) -> list[tuple[str, np.ndarray]]:
    """Return erosion layers from snapshot keys (same shape test as experiment_plots)."""
    out: list[tuple[str, np.ndarray]] = []
    for key in sorted(data.files):
        if not key.endswith("__0"):
            continue
        arr = data[key]
        if arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1 and arr.shape[2] == 1:
            display = key[: -len("__0")]
            out.append((display, arr))
    return out


def _normalize_axes_2d(axes: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    a = np.asarray(axes)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        if n_cols == 1:
            return a.reshape(-1, 1)
        if n_rows == 1:
            return a.reshape(1, -1)
    return a


def _values_minimal_vs_rest(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split flattened kernel values into Pareto-minimal vs non-minimal filters."""
    flat = weights[0, 0, 0, :, :]  # (patch, n_filters)
    _, pareto_mask = extract_pareto_filters(weights)
    minimal_vals = flat[:, pareto_mask].ravel()
    rest_vals = flat[:, ~pareto_mask].ravel()
    return minimal_vals, rest_vals, pareto_mask


def parse_weight_snapshot_plot_settings(train_cfg: dict) -> dict[str, Any]:
    """Defaults match ``run_experiment`` / ``base.yaml`` for snapshot plot subsampling."""
    if "weight_snapshot_plot_max_histograms" in train_cfg:
        mh = train_cfg["weight_snapshot_plot_max_histograms"]
        max_hist = None if mh is None else int(mh)
    else:
        max_hist = 10
    return {
        "max_histogram_snapshots": max_hist,
        "histogram_bins": int(train_cfg.get("weight_snapshot_histogram_bins", 40)),
    }


def select_snapshots_for_histogram(
    snapshots: list[dict],
    max_count: int | None,
) -> list[dict]:
    """Evenly subsample snapshot entries (by manifest order) for histogram plots."""
    if not snapshots:
        return []
    if max_count is None or max_count <= 0 or len(snapshots) <= max_count:
        return list(snapshots)
    n = len(snapshots)
    idx = np.linspace(0, n - 1, max_count, dtype=int)
    idx = np.unique(idx)
    return [snapshots[i] for i in idx]


def layer_keys_from_manifest(manifest_path: Path) -> list[str]:
    """Return sorted erosion layer keys (npz key prefixes) from the first readable snapshot."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)
    for snap in manifest.get("snapshots") or []:
        rel = snap.get("file")
        if not rel:
            continue
        npz_path = manifest_path.parent / rel
        if not npz_path.exists():
            continue
        data = _load_npz(npz_path)
        layers = _split_weights_npz_keys(data)
        return sorted(name for name, _ in layers)
    return []


def plot_weight_histogram_grid_by_layers(
    manifest_path: Path,
    layer_keys: list[str],
    snapshots_subset: list[dict],
    save_path: Path,
    plot_data_json: Path,
    bins: int = 40,
    plot_meta: dict | None = None,
) -> None:
    """Rows = SupErosion (erosion) layers; cols = training time snapshots.

    Each cell is **one** histogram of **all** scalar weights in that layer at that time,
    with Pareto-minimal vs non-minimal counts overlaid (red / blue), not one subplot per kernel.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() or not layer_keys or not snapshots_subset:
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_rows = len(layer_keys)
    n_cols = len(snapshots_subset)
    print(
        f"[plots] weight / histogram grid | {n_rows} layer row(s) × {n_cols} time column(s) "
        f"(all structuring elements pooled per cell)",
        flush=True,
    )

    # grid[i][j] = (minimal_vals, rest_vals) or None
    grid: list[list[tuple[np.ndarray, np.ndarray] | None]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]
    snap_meta: list[dict] = []

    for j, snap in enumerate(snapshots_subset):
        rel = snap.get("file")
        if not rel:
            continue
        npz_path = manifest_path.parent / rel
        if not npz_path.exists():
            continue
        ce = snap.get("completed_epochs")
        print(
            f"[plots] weight / histogram grid | column {j + 1}/{n_cols} | "
            f"completed_epochs={ce!r} | file={rel!r}",
            flush=True,
        )
        data = _load_npz(npz_path)
        layers = dict(_split_weights_npz_keys(data))
        for i, layer_key in enumerate(layer_keys):
            w = layers.get(layer_key)
            if w is None:
                continue
            min_v, rest_v, _ = _values_minimal_vs_rest(w)
            grid[i][j] = (min_v, rest_v)
        snap_meta.append(
            {
                "epoch_index": snap.get("epoch_index"),
                "completed_epochs": snap.get("completed_epochs"),
                "file": rel,
            }
        )

    flat_vals: list[np.ndarray] = []
    for row in grid:
        for cell in row:
            if cell is None:
                continue
            min_v, rest_v = cell
            flat_vals.append(np.r_[min_v, rest_v])
    if not flat_vals:
        print("[plots] weight / histogram grid | no data, skipping", flush=True)
        return

    all_vals = np.concatenate(flat_vals)
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    bin_edges = np.linspace(lo, hi, bins + 1)

    print(
        f"[plots] weight / histogram grid | building figure ({n_rows}×{n_cols}, bins={bins})",
        flush=True,
    )
    fig_w = max(2.8 * n_cols, 6.0)
    fig_h = max(2.8 * n_rows, 3.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = _normalize_axes_2d(axes, n_rows, n_cols)

    cells_json: list[dict] = []

    for i, layer_key in enumerate(layer_keys):
        for j in range(n_cols):
            ax = axes[i, j]
            cell = grid[i][j]
            if cell is None:
                ax.axis("off")
                continue
            min_v, rest_v = cell
            h_min, _ = np.histogram(min_v, bins=bin_edges)
            h_rest, _ = np.histogram(rest_v, bins=bin_edges)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(
                centers,
                h_rest,
                width=(bin_edges[1] - bin_edges[0]) * 0.92,
                align="center",
                color="blue",
                alpha=0.55,
                label="non-minimal",
            )
            ax.bar(
                centers,
                h_min,
                width=(bin_edges[1] - bin_edges[0]) * 0.92,
                align="center",
                color="red",
                alpha=0.55,
                label="Pareto minimal",
            )
            ce = snapshots_subset[j].get("completed_epochs", "")
            ax.set_title(f"epochs {ce}", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            if i == n_rows - 1:
                ax.set_xlabel("weight value", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{layer_key}\nfreq.", fontsize=8)
            if i == 0 and j == n_cols - 1:
                ax.legend(fontsize=6, loc="upper right")

            cells_json.append(
                {
                    "layer": layer_key,
                    "row": i,
                    "col": j,
                    "epoch_index": snapshots_subset[j].get("epoch_index"),
                    "completed_epochs": snapshots_subset[j].get("completed_epochs"),
                    "histogram_minimal": h_min.astype(int).tolist(),
                    "histogram_non_minimal": h_rest.astype(int).tolist(),
                    "histogram_bin_centers": centers.tolist(),
                    "n_minimal_scalar_weights": int(min_v.size),
                    "n_non_minimal_scalar_weights": int(rest_v.size),
                }
            )

    fig.suptitle(
        "Weight distributions (all structuring elements per layer; minimal vs rest overlaid)",
        fontsize=11,
    )
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[plots] weight / histogram grid | saving PNG → {save_path.name}", flush=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    payload = {
        "plot_type": "histogram_grid_layers_x_time",
        "layers": layer_keys,
        "histogram_bins": bins,
        "bin_edges": bin_edges.tolist(),
        "snapshot_columns": snap_meta,
        "cells": cells_json,
        "metadata": {
            "value_range": [lo, hi],
            "every_k_epochs": manifest.get("every_k_epochs"),
            "description": (
                "Each cell pools all kernel weights in that SupErosion layer at that time; "
                "red/blue split is Pareto-minimal vs non-minimal filter weights."
            ),
        },
    }
    if plot_meta:
        payload["plot_settings"] = plot_meta
    plot_data_json = Path(plot_data_json)
    plot_data_json.parent.mkdir(parents=True, exist_ok=True)
    with open(plot_data_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[plots] weight / histogram grid | saved plot data → {plot_data_json.name}", flush=True)


def generate_weight_evolution_artifacts(
    weight_snapshots_dir: Path | str,
    plots_dir: Path,
    plot_data_dir: Path,
    kernel_shape: tuple[int, int],
    histogram_bins: int = 40,
    *,
    max_histogram_snapshots: int | None = None,
) -> None:
    """If ``manifest.json`` exists under snapshots dir, write PNG + JSON plot data.

    One figure: **rows** = SupErosion (erosion) layers, **columns** = subsampled training times
    (default up to ``weight_snapshot_plot_max_histograms``). Each cell histograms **all** weights
    in that layer at that time.

    Args:
        max_histogram_snapshots: Max number of **time columns** (evenly spaced over saved snapshots).
            ``None`` or <=0 means use all snapshots (can be slow with many files).
    """
    weight_snapshots_dir = Path(weight_snapshots_dir)
    manifest_path = weight_snapshots_dir / "manifest.json"
    if not manifest_path.exists():
        return

    print(f"[plots] weight snapshots: reading manifest → {manifest_path}", flush=True)

    with open(manifest_path) as f:
        manifest = json.load(f)
    all_snapshots: list[dict] = list(manifest.get("snapshots") or [])
    if not all_snapshots:
        print("[plots] weight snapshots: manifest has no snapshots, skipping", flush=True)
        return

    hist_snaps = select_snapshots_for_histogram(all_snapshots, max_histogram_snapshots)

    print(
        f"[plots] weight snapshots: subsampling — manifest={len(all_snapshots)} → "
        f"time columns={len(hist_snaps)} (max_hist={max_histogram_snapshots!r})",
        flush=True,
    )

    plots_dir = Path(plots_dir)
    plot_data_dir = Path(plot_data_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    meta_hist = {
        "manifest_snapshots_total": len(all_snapshots),
        "max_histogram_snapshots": max_histogram_snapshots,
        "histogram_snapshots_used": len(hist_snaps),
        "kernel_shape": list(kernel_shape),
    }

    layer_keys = layer_keys_from_manifest(manifest_path)
    print(f"[plots] weight snapshots: layer rows (SupErosion blocks): {layer_keys!r}", flush=True)

    if not layer_keys:
        return

    plot_weight_histogram_grid_by_layers(
        manifest_path,
        layer_keys,
        hist_snaps,
        plots_dir / "weight_histogram_evolution_grid.png",
        plot_data_dir / "weight_histogram_evolution_grid.json",
        bins=histogram_bins,
        plot_meta=meta_hist,
    )
