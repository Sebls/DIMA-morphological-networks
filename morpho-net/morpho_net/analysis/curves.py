"""Training and validation loss curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_training_history(
    history: dict,
    output_path: str | Path,
    test_mse: float | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Save loss and metrics per epoch to a text file.

    Args:
        history: Keras history.history dict with keys like loss, val_loss, mse, val_mse.
        output_path: Path to save the .txt file.
        test_mse: Optional final test MSE (computed after training).
        elapsed_seconds: Optional total training time.
    """
    output_path = Path(output_path)
    lines = ["epoch\tloss\tval_loss\tmse\tval_mse"]
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    mse = history.get("mse", [])
    val_mse = history.get("val_mse", [])

    n_epochs = max(len(loss), len(val_loss), len(mse), len(val_mse))
    for e in range(n_epochs):
        l = float(loss[e]) if e < len(loss) else ""
        vl = float(val_loss[e]) if e < len(val_loss) else ""
        m = float(mse[e]) if e < len(mse) else ""
        vm = float(val_mse[e]) if e < len(val_mse) else ""
        lines.append(f"{e + 1}\t{l}\t{vl}\t{m}\t{vm}")

    if test_mse is not None or elapsed_seconds is not None:
        lines.append("")
        if elapsed_seconds is not None:
            lines.append(f"elapsed_seconds\t{elapsed_seconds}")
        if test_mse is not None:
            lines.append(f"test_mse\t{test_mse}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def plot_training_curves(
    history: dict,
    save_path: str | Path,
    title: str = "Training and Validation Loss",
    test_mse: float | None = None,
    log_scale: bool = False,
) -> None:
    """Plot training and validation loss across epochs.

    Args:
        history: Keras history.history dict.
        save_path: Path to save the figure.
        title: Plot title.
        test_mse: Optional final test MSE to show as horizontal reference line.
        log_scale: If True, use log scale for y-axis.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(loss) + 1)

    if loss:
        ax.plot(epochs, loss, "b-", label="Training loss", linewidth=1.5)
    if val_loss:
        ax.plot(epochs, val_loss, "r-", label="Validation loss", linewidth=1.5)
    if test_mse is not None:
        ax.axhline(y=test_mse, color="green", linestyle="--", alpha=0.7, label=f"Test MSE = {test_mse:.2e}")

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Loss (MSE) [log scale]", fontsize=12)
    else:
        ax.set_ylabel("Loss (MSE)", fontsize=12)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both" if log_scale else "major")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
