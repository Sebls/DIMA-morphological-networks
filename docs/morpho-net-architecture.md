# Research Framework Architecture (Simplified)

Think of the project as a **lightweight research framework for morphological neural networks**.
Each module isolates one responsibility so you can **iterate quickly on architectures, optimizers, and experiments** without unnecessary complexity.

The goal is:

- fast experimentation
- reproducibility
- modular components
- clean post-training analysis

---

# 1. config/

Purpose: **single experiment configuration (source of truth)**

This folder contains a **single `config.yaml`** used to define how experiments are executed.
Instead of managing multiple config files, you edit this file and it is **copied into each experiment folder at runtime**.

Typical contents:

- training hyperparameters
- dataset parameters
- model parameters
- initialization settings
- optimizer configuration
- logging options
- experiment metadata

Example sections:

- dataset (size, preprocessing, splits)
- model (type, kernel size, number of erosions)
- training (batch size, learning rate, epochs)
- initialization (ranges or strategies)
- seed and reproducibility settings

Practices:

- configurations are declarative (no Python logic)
- the config used for each run is **snapshotted in `experiments/`**
- experiments are fully reproducible from saved configs

Requirements:

- reproducibility via config snapshotting
- version control of the base config
- deterministic seeds when possible

---

# 2. scripts/

Purpose: **entry points for running the project**

This folder contains executable scripts that orchestrate experiments.

Typical scripts:

- `train.py` → main training pipeline
- `evaluate.py` → evaluation of trained models
- `run_experiment.py` → optional wrapper for automation

Responsibilities:

- load configuration
- create experiment directory
- copy config into experiment folder
- call dataset, model, and training modules

Practices:

- minimal logic
- delegate all work to `src/`
- CLI-friendly

---

# 3. src/data/

Purpose: **data loading and preprocessing**

This module prepares datasets for training.

Responsibilities:

- loading datasets (e.g., Fashion-MNIST)
- preprocessing (normalization, formatting)
- train/validation split
- batching

Structure:

- `dataset.py`
- `preprocessing.py`

Practices:

- no model logic
- deterministic behavior when seeded
- reusable pipelines across experiments

Requirements:

- consistent tensor shapes
- reproducible splits
- standardized normalization

---

# 4. src/layers/

Purpose: **core morphological operators (main research contribution)**

This is the **most important module**.

It contains reusable implementations of morphological layers.

Typical layers:

- erosion
- dilation
- supremum of erosions
- soft approximations

Structure:

- one file per operator
- each implemented as a reusable class

Practices:

- no experiment-specific logic
- framework-compliant (PyTorch / TensorFlow)
- document mathematical formulation
- clearly define input/output shapes

Requirements:

- differentiability
- correct gradient propagation
- compatibility with training pipeline
- support for serialization

---

# 5. src/models/

Purpose: **network architectures**

This module defines how layers are composed into models.

Typical architectures:

- single-layer SupErosion
- multi-layer networks
- receptive-field models

Structure:

- one file per architecture

Practices:

- keep architectures simple and readable
- no training logic inside models
- expose configurable parameters (kernel size, depth, etc.)

Requirements:

- consistent input/output shapes
- compatibility with training module
- easy model instantiation from config

---

# 6. src/training/

Purpose: **training logic (lightweight and flexible)**

This module manages model training and evaluation.

Contents:

- `trainer.py` → training loop
- `losses.py` → loss functions
- `metrics.py` → evaluation metrics

Responsibilities:

- training loop execution
- optimizer and loss setup
- metric computation
- checkpoint saving
- logging

Practices:

- independent from specific models
- configurable via config file
- reusable across experiments

Requirements:

- reproducible training
- consistent logging
- proper checkpointing

---

# 7. src/init/ (optional)

Purpose: **weight initialization strategies**

Morphological networks are sensitive to initialization, so this module provides reusable strategies.

Typical methods:

- random initialization
- patch-based initialization

Practices:

- independent of training logic
- reusable across models

Requirements:

- compatibility with layer shapes
- numerical stability

---

# 8. src/analysis/

Purpose: **post-training analysis and model comparison**

This module contains tools to analyze trained models and experiment results.

Current focus:

- Pareto frontier computation for:
  - complexity vs performance trade-offs
  - model simplification for inference
  - interpretability and comparison

Structure:

- `pareto.py`

Practices:

- operates only on saved experiment results
- not part of training pipeline
- lightweight and extensible

Requirements:

- correct multi-objective comparison
- numerical robustness

---

# 9. src/utils/

Purpose: **generic utilities**

Reusable helper functions used across the project.

Typical contents:

- `config.py` → config loading
- `logging.py` → logging utilities
- `plotting.py` → visualization (training curves, Pareto plots)

Practices:

- no domain-specific logic
- reusable across modules

---

# 10. experiments/

Purpose: **experiment tracking and reproducibility**

This folder stores outputs of all experiments.

Structure:

```bash
experiments/
└── exp_xxx/
    ├── config.yaml
    ├── metrics.json
    ├── model.pth
    └── logs/
```

Contents:

- frozen configuration used
- trained model
- evaluation metrics
- logs

Practices:

- one folder per run
- configs are copied automatically
- results are never overwritten

Requirements:

- full reproducibility
- easy comparison across experiments

---

# 11. logs/ and checkpoints/

Purpose: **runtime outputs**

- `logs/` → TensorBoard or console logs
- `checkpoints/` → optional shared checkpoints

These are auxiliary and not required for reproducibility (handled by `experiments/`).

---

# General Workflow

1. Edit `config/config.yaml`
2. Run training via script
3. Automatically create experiment folder
4. Save config, model, and metrics
5. Use `analysis/` for comparison (e.g., Pareto frontier)

---

# General Code Practices

### Modularity

Each module has a single responsibility.

### Reproducibility

Every experiment is reproducible from its saved config.

### Simplicity

Avoid unnecessary abstraction; prioritize iteration speed.

### Documentation

Each component should describe:

- inputs
- outputs
- tensor shapes
- mathematical behavior (for layers)

### Logging

Track:

- losses
- metrics
- configuration parameters

---

# Key Design Principle (important)

> Training, modeling, and analysis are clearly separated:
>
> - **layers/models** → define the system
> - **training** → optimize it
> - **analysis** → understand and compare it

