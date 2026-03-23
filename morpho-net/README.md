# morpho-net

Modular research framework for morphological neural networks. Refactored from the DIMA deep morphological networks experiments.

## Structure (per [morpho-net-architecture](../docs/morpho-net-architecture.md))

- **config/** — Experiment configurations (YAML, source of truth)
- **morpho_net/data/** — Data loading and preprocessing
- **morpho_net/layers/** — Morphological operators (dilation, erosion, SupErosions)
- **morpho_net/models/** — Network architectures
- **morpho_net/training/** — Training procedures and callbacks
- **morpho_net/analysis/** — Pareto frontier, kernel visualization
- **morpho_net/utils/** — Config loading, utilities
- **morpho_net/experiments/** — Reproducible experiment orchestration
- **experiments/** — Output folder (config snapshot, metrics, checkpoints)

## Usage

From the `morpho-net` directory:

```bash
# Install dependencies
uv sync

# Run experiment (config name or path; looks in config/ if relative)
uv run train --config quick_test.yaml
uv run train --config config/exp1_2pixel_single.yaml

# Evaluate trained model
uv run evaluate experiments/quick_test_001

# Alternative (full module path)
uv run python -m morpho_net.main --config quick_test.yaml
uv run python -m morpho_net.evaluate experiments/quick_test_001
```

