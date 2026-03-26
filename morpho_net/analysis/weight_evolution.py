"""Plots and saved plot data from weight snapshots across training."""

from __future__ import annotations

import json
from pathlib import Path

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


def plot_pareto_minimal_evolution(
    manifest_path: Path,
    kernel_shape: tuple[int, int],
    layer_key: str,
    save_path: Path,
    plot_data_npz: Path,
    plot_data_json: Path | None = None,
) -> None:
    """Rows = snapshot time; each row = Pareto-minimal structuring elements for that epoch."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    every_k = manifest.get("every_k_epochs")

    snapshots = manifest.get("snapshots") or []
    if not snapshots:
        return

    kernels_per_epoch: list[np.ndarray] = []
    epochs_meta: list[dict] = []

    for snap in snapshots:
        rel = snap.get("file")
        if not rel:
            continue
        npz_path = manifest_path.parent / rel
        if not npz_path.exists():
            continue
        data = _load_npz(npz_path)
        layers = dict(_split_weights_npz_keys(data))
        w = layers.get(layer_key)
        if w is None:
            continue
        pareto_filters, _ = extract_pareto_filters(w)
        n_p = pareto_filters.shape[1]
        if n_p == 0:
            continue
        kernels = pareto_filters.T.reshape(-1, *kernel_shape)
        kernels_per_epoch.append(kernels)
        epochs_meta.append(
            {
                "epoch_index": snap.get("epoch_index"),
                "completed_epochs": snap.get("completed_epochs"),
                "n_pareto": int(n_p),
            }
        )

    if not kernels_per_epoch:
        return

    n_rows = len(kernels_per_epoch)
    n_cols = max(k.shape[0] for k in kernels_per_epoch)

    vmin = min(float(np.min(k)) for k in kernels_per_epoch)
    vmax = max(float(np.max(k)) for k in kernels_per_epoch)

    fig_h = max(1.5 * n_rows, 2.0)
    fig_w = max(1.2 * n_cols, 1.1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w * n_cols, fig_h), squeeze=False)
    axes = _normalize_axes_2d(axes, n_rows, n_cols)
    fig.suptitle(
        f"Minimal structuring elements over training ({layer_key})",
        fontsize=12,
    )

    for r, kernels in enumerate(kernels_per_epoch):
        n_k = kernels.shape[0]
        for c in range(n_cols):
            ax = axes[r, c]
            if c < n_k:
                ax.imshow(
                    kernels[c],
                    cmap="gray",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
        if r < len(epochs_meta):
            ep = epochs_meta[r]
            ce = ep.get("completed_epochs", "")
            axes[r, 0].set_ylabel(
                f"epoch {ce}\n(n={ep.get('n_pareto', '')})",
                fontsize=8,
                rotation=0,
                labelpad=35,
                va="center",
            )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    plot_data_npz = Path(plot_data_npz)
    plot_data_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        plot_data_npz,
        layer=np.array(layer_key, dtype=object),
        kernel_shape=np.array(kernel_shape),
        epochs=np.array([m["completed_epochs"] for m in epochs_meta], dtype=np.int32),
        epoch_indices=np.array([m["epoch_index"] for m in epochs_meta], dtype=np.int32),
        n_pareto=np.array([m["n_pareto"] for m in epochs_meta], dtype=np.int32),
        kernels=np.array(kernels_per_epoch, dtype=object),
        vmin=np.array(vmin),
        vmax=np.array(vmax),
    )

    json_payload = {
        "layer": layer_key,
        "kernel_shape": list(kernel_shape),
        "every_k_epochs": every_k,
        "colormap_range": [vmin, vmax],
        "snapshots": epochs_meta,
        "kernels_npz": plot_data_npz.name,
        "description": (
            "kernels_npz contains array 'kernels' (object array): one (n_pareto, H, W) "
            "tensor per snapshot row in this figure."
        ),
    }
    if plot_data_json is not None:
        plot_data_json = Path(plot_data_json)
        plot_data_json.parent.mkdir(parents=True, exist_ok=True)
        with open(plot_data_json, "w") as f:
            json.dump(json_payload, f, indent=2)


def plot_weight_histogram_evolution(
    manifest_path: Path,
    layer_key: str,
    save_path: Path,
    plot_data_json: Path,
    bins: int = 40,
) -> None:
    """One row of histograms: red = Pareto-minimal filter weights, blue = others."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    snapshots = manifest.get("snapshots") or []
    if not snapshots:
        return

    series: list[tuple[dict, np.ndarray, np.ndarray]] = []

    for snap in snapshots:
        rel = snap.get("file")
        if not rel:
            continue
        npz_path = manifest_path.parent / rel
        if not npz_path.exists():
            continue
        data = _load_npz(npz_path)
        layers = dict(_split_weights_npz_keys(data))
        w = layers.get(layer_key)
        if w is None:
            continue
        min_v, rest_v, _ = _values_minimal_vs_rest(w)
        series.append((snap, min_v, rest_v))

    if not series:
        return

    all_vals = np.concatenate([np.r_[s[1], s[2]] for s in series])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    bin_edges = np.linspace(lo, hi, bins + 1)

    n_snap = len(series)
    fig, axes = plt.subplots(1, n_snap, figsize=(3.2 * n_snap, 3.5), squeeze=False)
    axes_flat = axes[0]

    json_snapshots: list[dict] = []

    for ax, (snap, min_v, rest_v) in zip(axes_flat, series):
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
        ce = snap.get("completed_epochs", "")
        ax.set_title(f"completed epochs {ce}", fontsize=9)
        ax.set_xlabel("weight value")
        ax.set_ylabel("frequency")
        ax.legend(fontsize=7)

        json_snapshots.append(
            {
                "epoch_index": snap.get("epoch_index"),
                "completed_epochs": snap.get("completed_epochs"),
                "bin_edges": bin_edges.tolist(),
                "histogram_minimal": h_min.astype(int).tolist(),
                "histogram_non_minimal": h_rest.astype(int).tolist(),
                "histogram_bin_centers": centers.tolist(),
                "count_minimal": int(np.sum(h_min)),
                "count_non_minimal": int(np.sum(h_rest)),
                "n_minimal_scalar_weights": int(min_v.size),
                "n_non_minimal_scalar_weights": int(rest_v.size),
            }
        )

    fig.suptitle(
        f"Weight distributions (minimal vs rest) — {layer_key}",
        fontsize=11,
    )
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    payload = {
        "layer": layer_key,
        "histogram_bins": bins,
        "bin_edges": bin_edges.tolist(),
        "snapshots": json_snapshots,
        "metadata": {
            "value_range": [lo, hi],
            "every_k_epochs": manifest.get("every_k_epochs"),
        },
    }
    plot_data_json = Path(plot_data_json)
    plot_data_json.parent.mkdir(parents=True, exist_ok=True)
    with open(plot_data_json, "w") as f:
        json.dump(payload, f, indent=2)


def generate_weight_evolution_artifacts(
    weight_snapshots_dir: Path | str,
    plots_dir: Path,
    plot_data_dir: Path,
    kernel_shape: tuple[int, int],
    histogram_bins: int = 40,
) -> None:
    """If ``manifest.json`` exists under snapshots dir, write PNGs and plot data files."""
    weight_snapshots_dir = Path(weight_snapshots_dir)
    manifest = weight_snapshots_dir / "manifest.json"
    if not manifest.exists():
        return

    plots_dir = Path(plots_dir)
    plot_data_dir = Path(plot_data_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    for layer_key in layer_keys_from_manifest(manifest):
        safe = layer_key.replace("/", "_")
        plot_pareto_minimal_evolution(
            manifest,
            kernel_shape,
            layer_key,
            plots_dir / f"pareto_minimal_evolution_{safe}.png",
            plot_data_dir / f"pareto_minimal_evolution_{safe}.npz",
            plot_data_json=plot_data_dir / f"pareto_minimal_evolution_{safe}.json",
        )
        plot_weight_histogram_evolution(
            manifest,
            layer_key,
            plots_dir / f"weight_histogram_evolution_{safe}.png",
            plot_data_dir / f"weight_histogram_evolution_{safe}.json",
            bins=histogram_bins,
        )
