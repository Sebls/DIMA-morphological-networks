# 1. Global Structure (High-Level Sections)

The notebook follows a **top-down research pipeline**:

```
1. Title + description
2. Mathematical background
3. Environment / setup
4. Dataset
5. Targets
6. Models
7. Training
8–9. Analysis & plots
10. Interpretation
11. Experiment driver (execution)
```

Each section is introduced with a **markdown cell using `##` headers**.

---

# 2. Cell Separation Pattern (Jupytext format)

Because it's `.py:percent`, cells are defined like:

### Code cell

```python
# %%
```

### Markdown cell

```python
# %% [markdown]
# ## Section title
# text...
```

### Markdown with ID (Colab/Jupyter metadata)

```python
# %% [markdown] id="abc123"
```

**Rule:**

* Every logical block = new `# %%`
* Every section = markdown cell first, then code

---

# 3. Section Design Pattern

Each section follows this **strict internal structure**:

### Pattern

```text
[Markdown: section title + explanation]

[Code: implementation]

[Markdown: explanation of code / formulas]

[Code: functions]
```

---

## Example pattern (reusable)

### Section header

```python
# %% [markdown]
# ## 5. Target definitions
```

### Theory (markdown)

```python
# %% [markdown]
# ### Horizontal map
# equation...
```

### Implementation

```python
# %%
def np_target_horizontal_diff_half(...):
```

This repeats everywhere:

* **math → explanation → code**

---

# 4. Markdown Style (VERY IMPORTANT)

This notebook uses **research-paper style markdown**:

### Features:

* LaTeX equations with `$$`
* Clear definitions
* Bullet explanations
* Section hierarchy:

```
## Section
### Subsection
#### Sub-subsection
```

---

## Example pattern:

```markdown
### 1.2 SIEM single-block map

We define:

$$
\psi_K(f)(x) = ...
$$
```

Key idea:

> Markdown is not comments — it's **formal documentation**

---

# 5. Code Organization Strategy

The code is grouped into **functional blocks**, not random cells.

---

## A. Utilities & helpers

* cropping
* shapes
* metrics

Always defined **before usage**

---

## B. Data pipeline

Functions like:

```python
resolve_data_pack()
load_bsds_pack()
```

Fully encapsulated → reusable

---

## C. Targets (VERY modular)

Pattern:

```python
def np_target_xxx(...)
```

* registry:

```python
TARGET_REGISTRY = {...}
```

This is **plug-and-play design**

---

## D. Models

Each model has:

```python
build_model_name(...)
```

And a **registry**:

```python
MODEL_REGISTRY = {...}
```

---

## E. Training pipeline

Central function:

```python
train_and_eval_suite(...)
```

This is the **engine of the notebook**

---

## F. Visualization

Separated cleanly:

* training curves
* predictions
* overlays

No mixing with training logic

---

# 6. Registry Pattern (VERY IMPORTANT)

This notebook uses **registries everywhere**:

### Targets

```python
TARGET_REGISTRY = {...}
```

### Models

```python
MODEL_REGISTRY = {...}
```

### Labels

```python
TARGET_LABELS = {...}
```

This allows:

* dynamic selection
* clean experiments
* no hardcoding

---

# 7. Hyperparameters Block (Centralized)

Single cell:

```python
# --- Hyperparameters ---
RANDOM_SEED = 42
EPOCHS_MAIN = 500
BATCH_SIZE = 2
...
```

Rule:

> All tunable parameters in ONE place

---

# 8. Experiment Driver Pattern

Main function:

```python
def run_experiment(target_key, ...):
```

### Responsibilities:

1. create run folder
2. load data
3. compute targets
4. train models
5. save results
6. plot
7. log metadata

This is the **orchestrator**

---

# 9. Execution Cells (Final Section)

Each experiment = one cell:

```python
suite_mg9 = run_experiment("mg9", ...)
```

Pattern:

* One cell per experiment
* No logic inside → just call

---

# 10. Output Structure (Reproducibility)

Each run creates:

```
outputs/
 └── exp_xxx_timestamp/
      ├── plots/
      ├── checkpoints/
      ├── metrics.json
      └── metadata.json
```

This is **experiment tracking built-in**

---

# 11. Clean Separation of Concerns

The notebook enforces:

| Layer     | Responsibility |
| --------- | -------------- |
| Markdown  | theory         |
| Functions | logic          |
| Registry  | configuration  |
| Driver    | execution      |
| Cells     | orchestration  |

---

# 12. Key Design Philosophy

This notebook is built like:

> **A research framework, not a notebook**

Core principles:

* modular
* reproducible
* configurable
* theory + code aligned
* minimal duplication

---

# 13. Template You Should Reuse

If you create a new notebook, follow this skeleton:

```python
# %% [markdown]
# Title + description

# %% [markdown]
# ## 1. Theory

# %% [markdown]
# explanation

# %%
# implementation

# %% [markdown]
# ## 2. Data

# %%
# data functions

# %% [markdown]
# ## 3. Targets

# %%
# target functions + registry

# %% [markdown]
# ## 4. Models

# %%
# model definitions + registry

# %% [markdown]
# ## 5. Training

# %%
# training functions

# %% [markdown]
# ## 6. Experiments

# %%
run_experiment(...)
```