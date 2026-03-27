"""Generate all experiment plots: training curves, structuring elements, Pareto minimals."""

from __future__ import annotations

from pathlib import Path

from morpho_net.analysis.curves import save_training_history, plot_training_curves
from morpho_net.analysis.kernels import plot_structuring_elements, plot_pareto_elements
from morpho_net.analysis.pareto import extract_pareto_filters
from morpho_net.analysis.weight_evolution import generate_weight_evolution_artifacts


def _get_erosion_layers(model):
    """Find all MorphologicalDilation (erosion) layers in the model."""
    erosion_layers = []
    for layer in model.layers:
        if hasattr(layer, "weights") and layer.weights:
            w = layer.get_weights()
            if w and len(w) > 0:
                shape = w[0].shape
                # MorphologicalDilation: (1, 1, 1, patch_size, filters)
                if len(shape) == 5 and shape[0] == 1 and shape[1] == 1 and shape[2] == 1:
                    erosion_layers.append((layer.name, w[0]))
    return erosion_layers


def generate_experiment_plots(
    model,
    history,
    output_dir: str | Path,
    kernel_shape: tuple[int, int] = (3, 3),
    n_show: int = 100,
    test_mse: float | None = None,
    elapsed_seconds: float | None = None,
    checkpoint_dir: str | Path | None = None,
    weight_snapshot_histogram_bins: int = 40,
    weight_snapshot_plot_max_histograms: int | None = 10,
) -> None:
    """Generate all plots for an experiment: loss curves, structuring elements, Pareto minimals.

    Args:
        model: Trained Keras model.
        history: Keras History object (history.history).
        output_dir: Directory to save plots and loss_history.txt.
        kernel_shape: (H, W) of each kernel.
        n_show: Max number of structuring elements to show in "all" plot (~100).
        test_mse: Test MSE if available.
        elapsed_seconds: Wall time for training if available.
        checkpoint_dir: Checkpoint dir (contains ``weight_snapshots/`` when snapshots enabled).
        weight_snapshot_histogram_bins: Bins for weight-distribution evolution plot data.
        weight_snapshot_plot_max_histograms: Max **time columns** in the layer×time histogram grid;
            ``None`` = all saved snapshots (can be slow).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save loss history to .txt
    print("[plots] step: loss history + training/validation curves (linear + log)", flush=True)
    hist_data = history.history if hasattr(history, "history") else history
    save_training_history(
        hist_data,
        output_dir / "loss_history.txt",
        test_mse=test_mse,
        elapsed_seconds=elapsed_seconds,
    )

    # 2. Training/validation/test loss curves (linear and log scale)
    plot_training_curves(
        hist_data,
        plots_dir / "training_curves.png",
        title="Training and Validation Loss",
        test_mse=test_mse,
        log_scale=False,
    )
    plot_training_curves(
        hist_data,
        plots_dir / "training_curves_log.png",
        title="Training and Validation Loss (log scale)",
        test_mse=test_mse,
        log_scale=True,
    )

    # 3. Structuring elements from each MorphologicalDilation layer
    print("[plots] step: structuring elements (all kernels + Pareto minimal per layer)", flush=True)
    erosion_layers = _get_erosion_layers(model)
    for layer_name, weights in erosion_layers:
        safe_name = layer_name.replace("/", "_")
        n_total = weights.shape[-1]

        # 3a. All elements (limited to n_show)
        n_displayed = min(n_show, n_total)
        print(f"  … structuring_elements all: {layer_name!r} ({n_displayed} of {n_total})", flush=True)
        plot_structuring_elements(
            weights,
            kernel_shape=kernel_shape,
            n_show=n_displayed,
            cols=10,
            title=f"Structuring Elements ({layer_name}) — {n_displayed} of {n_total}",
            save_path=plots_dir / f"structuring_elements_{safe_name}_all.png",
            show=False,
        )

        # 3b. Pareto-minimal elements (no limit)
        try:
            pareto_filters, _ = extract_pareto_filters(weights)
            n_pareto = pareto_filters.shape[1]
            if pareto_filters.size > 0:
                print(f"  … structuring_elements pareto: {layer_name!r} (n_pareto={n_pareto})", flush=True)
                plot_pareto_elements(
                    pareto_filters,
                    kernel_shape=kernel_shape,
                    title=f"Minimal Structuring Elements - Pareto ({layer_name}) — {n_pareto} elements",
                    save_path=plots_dir / f"structuring_elements_{safe_name}_pareto.png",
                    show=False,
                )
        except Exception:
            pass  # Skip if Pareto extraction fails

    # 4. Weight snapshots: Pareto evolution + histogram row (uses Checkpoint/weight_snapshots/)
    if checkpoint_dir is not None:
        print(
            "[plots] step: weight snapshots — layer×time histogram grid (if manifest exists)",
            flush=True,
        )
        plot_data_dir = output_dir / "plot_data"
        generate_weight_evolution_artifacts(
            Path(checkpoint_dir) / "weight_snapshots",
            plots_dir,
            plot_data_dir,
            kernel_shape=kernel_shape,
            histogram_bins=weight_snapshot_histogram_bins,
            max_histogram_snapshots=weight_snapshot_plot_max_histograms,
        )
    print("[plots] finished", flush=True)
