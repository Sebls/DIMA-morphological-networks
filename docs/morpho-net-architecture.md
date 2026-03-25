# Research framework architecture (morpho-net)

Lightweight layout for **morphological neural network** experiments: clear split between **config**, **model definition**, **training strategy**, and **analysis**, with reproducibility via snapshotted YAML.

Design goals:

- fast experimentation (swap architectures, initializers, training procedures from config)
- reproducibility (frozen config per run)
- modular components (registries for new methods without editing the experiment runner)
- post-training analysis (Pareto, kernel plots)

---

## 1. `morpho-net/config/`

**Purpose:** experiment definitions (source of truth).

- YAML files per experiment; **`extends`** merges a base (e.g. `base.yaml`) via `morpho_net.utils.config.load_config`.
- Typical sections: `dataset`, `ground_truth`, `model`, `initialization`, `training`, `callbacks`, `output`, `experiment`.
- Each run copies the **resolved** config into the experiment output folder.

Practices: declarative only; seeds and paths live in YAML where possible.

---

## 2. `morpho-net/morpho_net/` — Python package

### Entry points

- **`main.py`** — CLI: `uv run train --config …` (see project `pyproject.toml` scripts).
- **`evaluate.py`** — load `config.yaml` + weights from an experiment directory and report metrics.

### `experiments/`

- **`run_experiment.py`** — orchestration: load config → data → ground truth → `build_model` → `run_training` → metrics, plots, checkpoints.

---

## 3. `morpho_net/data/`

**Purpose:** data loading and targets.

- Dataset loading (e.g. Fashion-MNIST with optional noise), splits.
- Ground-truth filter application for supervised targets.

No model or training-loop logic here.

---

## 4. `morpho_net/layers/`

**Purpose:** morphological building blocks (Keras layers).

- **`dilation.py`** — `MorphologicalDilation` (learnable flat structuring elements on patches).
- **`sup_erosions.py`** — supremum-of-erosions blocks (single- and two-input variants).
- **`erosion.py`** — related helpers as needed.

Layers accept optional **Keras initializers** (from `initialization`) for weights; keep math and shapes documented in code.

---

## 5. `morpho_net/models/`

**Purpose:** full architectures composed from layers.

- One module per architecture (e.g. single-layer SupErosions, two-layer, receptive-field variant).
- **`registry.py`** — maps config `model.architecture` string → `build_from_config(model_cfg, init_cfg)`.

Models stay free of training logic; parameters come from YAML.

---

## 6. `morpho_net/initialization/`

**Purpose:** **config-driven weight initialization** (extensible registry).

- **`registry.py`** — `build_kernel_initializer(merged_cfg)` keyed by `method` or legacy `strategy` (`uniform`, `random_e_patches`, `minimal_e_patches`, …).
- **`base.py`** — shared `MorphoInitializer` pattern.
- Concrete initializers in dedicated modules; register new names with `register_initializer` or by extending `INITIALIZER_REGISTRY`.

Per-block options for multi-branch models are merged in **`morpho_net.utils.merge.merge_init_block`** (`block1` / `block2` / `block3` overrides on top of global `initialization`).

Initializers are **`@keras.saving.register_keras_serializable`** where needed for layer `get_config` / deserialization.

---

## 7. `morpho_net/training/`

**Purpose:** compile, fit, callbacks, and **pluggable training procedures**.

| Piece | Role |
|--------|------|
| **`fit.py`** | `compile_model`, `train_model` (Keras `fit`, checkpoint + early-stop callbacks). |
| **`callbacks.py`** | Shared callback factory (checkpoint path, loss threshold stop). |
| **`train.py`** | `run_training` — dispatches on config. |
| **`registry.py`** | Maps `training.update_method` (or `training.method`) → `TrainingProcedure` subclass. |
| **`procedures/`** | One module per procedure: e.g. `StandardFitProcedure`, `AlphaSchedulerProcedure`; stubs for future custom loops (`pareto_update`, `dense_update`, `soft_sup_erosions`). |

Add a new training mode by subclassing **`procedures/base.TrainingProcedure`** and registering it with **`register_training_procedure`**.

---

## 8. `morpho_net/analysis/`

**Purpose:** post-training understanding (not required during `fit`).

- **`pareto.py`** — Pareto-efficient structuring elements.
- **`experiment_plots.py`**, **`kernels.py`**, **`curves.py`** — plots and diagnostics tied to experiment outputs.

---

## 9. `morpho_net/utils/`

**Purpose:** cross-cutting helpers.

- **`config.py`** — YAML load + `extends` + `deep_merge`.
- **`merge.py`** — `deep_merge`, `merge_init_block` for nested dicts / initialization blocks.
- **`experiment.py`** — experiment directory naming / sequencing.

---

## 10. `experiments/` (output root)

**Purpose:** artifacts per run (under `morpho-net/` or path from config `output.results_dir`).

Typical contents per run:

- `config.yaml` — frozen resolved config
- `metrics.json`
- `Checkpoint/` — best weights (e.g. `checkpoint.weights.h5`)
- plots from analysis pipeline

One subdirectory per experiment name + run index; configs are not overwritten across repeats.

---

## Workflow

1. Add or edit YAML under **`morpho-net/config/`** (optionally `extends: base.yaml`).
2. Run **`uv run train --config <file>`** from the `morpho-net` directory.
3. Inspect **`experiments/<name>_###/`** for config snapshot, metrics, checkpoints, plots.
4. Use **`uv run evaluate <experiment_dir>`** for fresh metrics on saved weights.

---

## Design principle

> **Layers / models** define the forward map.  
> **Initialization** sets starting weights from config.  
> **Training** (procedure + `fit`) optimizes them.  
> **Analysis** interprets and compares runs.

Keep those boundaries when adding features.
