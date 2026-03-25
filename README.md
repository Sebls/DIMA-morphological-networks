# morpho-net

Modular research framework for morphological neural networks. Refactored from the DIMA deep morphological networks experiments.

## Structure (see [docs/morpho-net-architecture.md](docs/morpho-net-architecture.md))

Paths are under **`morpho-net/`** unless noted.

- **config/** — Experiment YAML (`extends` for shared defaults); source of truth per run
- **morpho_net/data/** — Datasets, splits, ground-truth targets
- **morpho_net/layers/** — Keras morphological layers (dilation, SupErosions, …)
- **morpho_net/models/** — Architectures + **registry** mapping `model.architecture` → builder
- **morpho_net/initialization/** — Config-driven weight init (**registry**: `method` / `strategy`; per-block `block1`…`block3`)
- **morpho_net/training/** — `fit.py` (compile + fit), **callbacks**, **`run_training`** dispatcher, **registry** + **procedures/** (`update_method`: standard fit, alpha scheduler, future custom loops)
- **morpho_net/analysis/** — Pareto filters, kernel plots, curves
- **morpho_net/utils/** — Config load/merge, **merge_init_block**, experiment paths
- **morpho_net/experiments/** — `run_experiment` orchestration; **main.py** / **evaluate.py** CLI
- **experiments/** — Run outputs (snapshotted config, metrics, checkpoints, plots)

## Usage

From the **`morpho-net/`** directory (or any cwd where the package is on the path). Console scripts are defined in `pyproject.toml`:

| Script | Entry point |
|--------|-------------|
| `train` | `morpho_net.main:main` |
| `evaluate` | `morpho_net.evaluate:main` |
| `morpho-train` | same as `train` |
| `morpho-evaluate` | same as `evaluate` |

```bash
# Install dependencies
uv sync

# Train (required: --config / -c)
uv run train --config quick_test.yaml
uv run train -c config/exp1_2pixel_single.yaml
uv run morpho-train --config quick_test.yaml --output-dir /tmp/my_run --seed 42

# Optional train flags:
#   --output-dir, -o   Output directory (default: experiments/<name> from config)
#   --seed             Global NumPy/TF seed for model init and training (default: 42);
#                      dataset noise still uses dataset.seed in the YAML

# Evaluate: positional experiment directory (relative to cwd; needs config.yaml, Checkpoint/)
uv run evaluate experiments/quick_test_001
uv run morpho-evaluate experiments/quick_test_001 --seed 42

# Optional evaluate flag:
#   --seed   Seed for dataset noise only (default: 42)

# Same CLIs via module path
uv run python -m morpho_net.main --config quick_test.yaml
uv run python -m morpho_net.evaluate experiments/quick_test_001
```

Relative config paths are resolved against the package root then `config/` (see `morpho_net.main`).

