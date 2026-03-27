"""Run a single experiment from configuration."""

from __future__ import annotations

import json
import yaml
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

from morpho_net.utils.config import load_config
from morpho_net.utils.experiment import next_experiment_dir
from morpho_net.data import load_fashion_mnist_noisy, create_ground_truth_model, generate_ground_truth
from morpho_net.models import build_model
from morpho_net.training import run_training
from morpho_net.analysis.experiment_plots import generate_experiment_plots
from morpho_net.analysis.weight_evolution import parse_weight_snapshot_plot_settings


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Run experiment from config file.

    ``seed`` sets the global NumPy and TensorFlow RNGs (model init, training).
    Dataset Gaussian noise uses ``config["dataset"]["seed"]`` only, via a
    dedicated :class:`numpy.random.Generator`, so it does not clobber this
    global state.

    Returns dict with metrics, history, and paths.
    """
    config = load_config(config_path)
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # Output directory (experiments/ per architecture doc)
    # Sequential index allows running same experiment multiple times without overwriting
    exp_name = config.get("experiment", {}).get("name", "experiment")
    results_base = Path(config.get("output", {}).get("results_dir", "experiments"))
    if output_dir is None:
        output_dir = next_experiment_dir(results_base, exp_name)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / config.get("output", {}).get("checkpoint_dir", "Checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy (snapshot for reproducibility)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Dataset
    ds_cfg = config.get("dataset", {})
    train_split, val_split, test_split = load_fashion_mnist_noisy(
        size=ds_cfg.get("size", 100),
        use_noise=ds_cfg.get("use_noise", True),
        sigma=ds_cfg.get("sigma", 40),
        seed=ds_cfg.get("seed", 42),
    )

    # Ground truth
    gt_kernel = config.get("ground_truth", {}).get("kernel", [[0, 0.5, 0], [0, 0.5, 0], [0, 0, 0]])
    gt_model = create_ground_truth_model(gt_kernel)
    y_train, y_val, y_test = generate_ground_truth(
        gt_model,
        train_split.images,
        val_split.images,
        test_split.images,
    )

    # Build model
    model_cfg = config.get("model", {})
    arch = model_cfg.get("architecture", "single_sup_erosions")
    init_cfg = config.get("initialization", {})
    model = build_model(arch, model_cfg, init_cfg)

    # Merge output config for checkpoint path
    merged_config = {**config, "output": {**config.get("output", {}), "checkpoint_dir": str(checkpoint_dir)}}

    # Train model
    start = time.perf_counter()
    history = run_training(
        model,
        train_split.images,
        y_train,
        val_split.images,
        y_val,
        merged_config,
        output_dir=checkpoint_dir,
    )
    elapsed = time.perf_counter() - start

    # Load best weights
    best_path = checkpoint_dir / "checkpoint.weights.h5"
    if best_path.exists():
        model.load_weights(str(best_path))

    # Evaluate
    mse_fn = MeanSquaredError()
    y_val_pred = model.predict(val_split.images, verbose=0)
    y_test_pred = model.predict(test_split.images, verbose=0)

    if y_val_pred.shape[-1] == 1:
        y_val_pred = y_val_pred[:, :, :, 0]
    if y_test_pred.shape[-1] == 1:
        y_test_pred = y_test_pred[:, :, :, 0]

    val_mse = float(mse_fn(y_val, y_val_pred).numpy())
    test_mse = float(mse_fn(y_test, y_test_pred).numpy())

    # Results
    n_epochs = len(history.history.get("loss", []))
    results = {
        "experiment": exp_name,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "n_epochs": n_epochs,
        "elapsed_seconds": elapsed,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate plots: loss curves, structuring elements (all + Pareto minimal), weight snapshots
    kernel_shape = tuple(model_cfg.get("kernel_size", [3, 3]))
    train_cfg = config.get("training", {})
    wplot = parse_weight_snapshot_plot_settings(train_cfg)
    generate_experiment_plots(
        model,
        history,
        output_dir,
        kernel_shape=kernel_shape,
        n_show=100,
        test_mse=test_mse,
        elapsed_seconds=elapsed,
        checkpoint_dir=checkpoint_dir,
        weight_snapshot_histogram_bins=wplot["histogram_bins"],
        weight_snapshot_plot_max_histograms=wplot["max_histogram_snapshots"],
    )

    return results
