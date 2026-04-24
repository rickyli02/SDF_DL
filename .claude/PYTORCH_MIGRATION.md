# PyTorch Migration Plan

## Overview

Migrate the TensorFlow 1.x (via `tf.compat.v1`) codebase to PyTorch, updating Python to 3.12.
The original code is moved to `tf_reference/` unchanged; the rewrite starts from scratch using
the reference as a spec. Config files, datasets, notebooks (structure), and non-model utilities
are copied directly.

---

## Phase 0 — Pre-Migration Checklist

Complete every item here before touching any code. These are the acceptance criteria for the
migration's correctness.

### 0.1 Environment snapshot

- [ ] Record current Python version: `python --version`
- [ ] Freeze current deps: `pip freeze > .claude/tf_env_freeze.txt`
- [ ] Confirm `.venv/` is excluded from git (already gitignored)

### 0.2 Baseline numerical outputs (golden values)

Run the following using the **existing TF codebase + sample checkpoints** and save outputs as
`.npy` files in `.claude/golden/`. These become regression targets for the PyTorch port.

| Script / step | What to capture | Output file |
|---|---|---|
| `run.py` (1 epoch, fixed seed) | Sharpe per epoch, mean SDF weight | `golden/sdf_epoch1.npy` |
| `model_GAN.py → getSDF()` | SDF array on test split | `golden/sdf_test.npy` |
| `model_GAN.py → getWeightWithData()` | Weight vector on test split | `golden/weights_test.npy` |
| `run_RtnFcst_ensembles.py` (1 epoch) | Loss curve | `golden/rtnfcst_epoch1.npy` |

Capture script skeleton:
```python
# .claude/capture_golden.py
import numpy as np, json, os
# load sample_checkpoints model, run inference on Char_test / macro_test
# np.save('.claude/golden/sdf_test.npy', sdf_array)
```

> Golden values only need to be **close**, not bit-exact (different RNG/precision).
> Acceptance threshold: Sharpe within ±0.05, weight correlation ≥ 0.99.

### 0.3 Smoke tests (TF baseline must pass)

Write these in `tests/` **before** migration so they can be re-run against the PyTorch port.

```
tests/
  conftest.py            # shared fixtures: tiny synthetic dataset
  test_data_layer.py     # DataInRamInputLayer iteration, shapes, mask logic
  test_model_shapes.py   # forward pass output shapes (GAN, RtnFcst)
  test_training_step.py  # one optimizer step doesn't NaN
  test_sdf_properties.py # SDF mean ≈ 1, weights sum-to-zero per period
  test_golden.py         # compare PyTorch outputs to .claude/golden/ files
```

Synthetic dataset fixture (no real data needed):
```python
# tests/conftest.py
import numpy as np, pytest

@pytest.fixture
def tiny_dataset(tmp_path):
    T, N, F_char, F_macro = 10, 20, 46, 178
    np.savez(tmp_path / 'char.npz',
        data=np.random.randn(T, N, F_char).astype('float32'),
        mask=np.ones((T, N), dtype=bool),
        ret=np.random.randn(T, N).astype('float32'))
    np.savez(tmp_path / 'macro.npz',
        data=np.random.randn(T, F_macro).astype('float32'))
    return tmp_path
```

- [ ] All `tests/test_data_layer.py` and `tests/test_model_shapes.py` pass against TF code
- [ ] `tests/test_golden.py` is written with tolerance assertions
- [ ] `pytest tests/ -x` exits 0 on the TF codebase

### 0.4 Git hygiene

- [ ] Create branch: `git checkout -b pytorch-migration`
- [ ] Confirm `sample_checkpoints/` is committed (reference weights for comparison)
- [ ] Add `tf_reference/` and `output*/` to `.gitignore` if not already

---

## Phase 1 — Freeze the Reference

```bash
mkdir tf_reference
cp -r src config config_RF run.py run_RtnFcst_ensembles.py create_RF_data.py \
       requirements.txt tf_reference/
echo "# TF1 reference — do not edit" > tf_reference/README.md
git add tf_reference/ && git commit -m "chore: freeze TF1 reference before PyTorch rewrite"
```

Resulting layout after Phase 1:
```
tf_reference/               ← original code, frozen
  src/
  config/
  config_RF/
  run.py  run_RtnFcst_ensembles.py  create_RF_data.py
  requirements.txt
src/                        ← will be rewritten in Phase 2
...
```

---

## Phase 2 — New Environment

### 2.1 Python version

Upgrade to Python 3.12 (PyTorch ≥ 2.3 supports 3.12 on Apple Silicon).

```bash
# with pyenv
pyenv install 3.12.x
pyenv local 3.12.x
python -m venv .venv312 && source .venv312/bin/activate
```

### 2.2 New `requirements.txt`

Replace the TF requirements file entirely:

```text
# Core
torch==2.3.*
torchvision==0.18.*          # for MPS/CUDA utilities
numpy==1.26.*
pandas==2.2.*
scipy==1.13.*
scikit-learn==1.5.*

# Training utilities
tensorboard==2.17.*          # replaces tf.summary

# Analysis / notebook
matplotlib==3.9.*
seaborn==0.13.*
statsmodels==0.14.*
notebook==7.2.*
jupyterlab==4.*

# Testing
pytest==8.*
pytest-cov==5.*
```

- [ ] `pip install -r requirements.txt` succeeds
- [ ] `python -c "import torch; print(torch.backends.mps.is_available())"` prints `True` on Apple Silicon

---

## Phase 3 — File-by-File Rewrite

### Files copied directly (no changes needed)

| File | Action |
|---|---|
| `config/config.json` | copy as-is |
| `config_RF/config_RF_1.json` | copy as-is |
| `datasets/` | gitignored, untouched |
| `src/utils.py` | copy — `deco_print`, `sharpe` are framework-agnostic |
| `README.md` | update env/install section only |
| `.claude/CLAUDE.md` | update after migration complete |

### Files deleted (TF-only, no PyTorch equivalent needed)

| File | Reason |
|---|---|
| `src/tf_compat.py` | entire purpose was TF1→TF2 shim |

### Files rewritten

#### 3.1 `src/data/data_layer.py` — **low complexity, start here**

TF dependency: none (pure NumPy). The class `DataInRamInputLayer` and its `iterateOneEpoch`
iterator are already framework-agnostic. Copy with minimal changes:
- Replace any `tf.*` dtype references with `np.*`
- Optionally add a `torch.utils.data.Dataset` wrapper for the training loop

Mapping:
```
DataInRamInputLayer.__init__    → copy verbatim
DataInRamInputLayer.iterateOneEpoch → copy verbatim
(new) CharMacroDataset(Dataset) → wraps iterateOneEpoch for DataLoader
```

#### 3.2 `src/model/model_utils.py` — **low complexity**

Contains `create_rnn_cell`, `initial_state_size`, `calculateStatistics`.
- `create_rnn_cell` → return `torch.nn.LSTM` / `torch.nn.GRU` / `torch.nn.RNN`
- `initial_state_size` → return `(num_layers, batch, hidden)` tuple convention
- `calculateStatistics` → pure NumPy, copy verbatim

#### 3.3 `src/model/model_base.py` — **medium complexity**

TF1 patterns to replace:

| TF1 pattern | PyTorch equivalent |
|---|---|
| `tf.placeholder` | method arguments / `torch.Tensor` |
| `sess.run(...)` | `model.forward(...)` |
| `tf.train.Saver` | `torch.save(model.state_dict(), path)` |
| `tf.train.latest_checkpoint` | `glob` + sort on `*.pt` files |
| `tf.global_variables_initializer` | default random init (no-op) |
| `tf.train.AdamOptimizer` | `torch.optim.Adam` |
| `tf.train.MomentumOptimizer` | `torch.optim.SGD(momentum=...)` |
| `tf.train.exponential_decay` | `torch.optim.lr_scheduler.ExponentialLR` |
| `self._global_step` | `self.global_step: int` counter |

New base class interface:
```python
class ModelBase(nn.Module):
    def __init__(self, model_params, mode):  # drop global_step arg
    def build_optimizer(self) -> torch.optim.Optimizer
    def save(self, logdir: str, step: int) -> None
    def load(self, logdir: str) -> None       # loads latest *.pt
    def forward(self, I_macro, I, R, mask) -> dict  # replaces sess.run
```

#### 3.4 `src/model/model_GAN.py` — **high complexity, core rewrite**

This is the largest file (649 lines). Work section by section:

**3.4a — SDF network (the "generator")**

```
TF: Dense + dropout + LSTM via MultiRNNCell + dynamic_rnn
PT: nn.Linear + nn.Dropout + nn.LSTM(batch_first=True)
```

Key architectural points (preserve these exactly):
- Firm characteristics `I` → FF layers → firm-level score
- Macro features `I_macro` → LSTM → time-varying state
- SDF weight = softmax(score) per time period
- SDF = 1 - Σ(weight_i × return_i)

**3.4b — Moment network (the "discriminator")**
- Same FF → LSTM structure but outputs `num_condition_moment` factors
- Adversarial loss: `E[SDF × factor_return]² - λ × Sharpe(SDF)²`

**3.4c — Training loop**

TF1 alternating session-based training → PyTorch training loop:
```python
for epoch in range(num_epochs):
    for I_macro, I, R, mask in dataloader:
        # step 1: update moment network
        opt_moment.zero_grad()
        loss_moment = -gan_loss(...)
        loss_moment.backward(); opt_moment.step()
        # step 2: update SDF network
        opt_sdf.zero_grad()
        loss_sdf = gan_loss(...)
        loss_sdf.backward(); opt_sdf.step()
```

**3.4d — Inference methods** (keep same public API for notebook compatibility)

```python
def getWeightWithData(self, dl, initial_state=None, normalized=False) -> np.ndarray
def getSDF(self, dl, initial_state=None) -> np.ndarray
def getNormalizedSDF(self, dl, initial_state=None) -> np.ndarray
def getSDFFactor(self, dl, initial_state=None) -> np.ndarray
def calculateStatistics(self, dl, initial_state=None) -> dict
```
Drop `sess` parameter from all signatures (was only needed for TF1 graph execution).

**3.4e — Ensemble class**

`FeedForwardModelWithNA_GAN_Ensembled` — mostly orchestration, copy structure,
drop `sess` args throughout.

#### 3.5 `src/model/model_RtnFcst.py` — **medium complexity**

236 lines. Similar FF + LSTM structure to GAN but supervised regression loss.
- Same `nn.Module` pattern
- MSE / rank-IC loss depending on config
- Drop `sess` arg from all public methods

#### 3.6 `run.py` — **low complexity after model rewrite**

Remove:
- `tf.Session()` context manager
- `sess.run(tf.global_variables_initializer())`
- `summary_writer` / `tf.summary.FileWriter`

Add:
- `torch.manual_seed(seed)` for reproducibility
- `SummaryWriter` from `torch.utils.tensorboard`
- MPS device selection: `device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`

#### 3.7 `run_RtnFcst_ensembles.py` — **low complexity after model rewrite**

Same changes as `run.py`. Ensemble loop structure stays the same.

#### 3.8 `create_RF_data.py` — **copy with minor edits**

Calls `model.getSDF()` / `model.getWeightWithData()` — update signatures to drop `sess`.
Otherwise pure NumPy/pandas, copy verbatim.

#### 3.9 `model_GAN.ipynb` — **update imports and cell outputs only**

- Replace `from src.tf_compat import tf` with `import torch`
- Update model instantiation calls (drop `sess`, `tSize` constructor arg style)
- Re-run cells to regenerate outputs after rewrite is complete

---

## Phase 4 — Checkpoint Conversion (optional)

The `sample_checkpoints/` weights are TF1 format. If you need to initialize the PyTorch model
from them (rather than retraining from scratch):

1. Load TF checkpoint in a minimal TF environment: `tf.train.load_checkpoint(path)`
2. Dump each variable to `.npy`
3. Build a name-mapping dict from TF variable names → PyTorch `state_dict` keys
4. Load via `model.load_state_dict(mapped_dict)`

This is optional — the paper results can be reproduced by retraining. Only needed if you want
the exact pretrained weights.

---

## Phase 5 — Validation Gates

After each 3.x subsection, run the relevant test before moving on.

| Gate | Command | Must pass before |
|---|---|---|
| G1: data layer | `pytest tests/test_data_layer.py` | starting 3.3 |
| G2: model shapes | `pytest tests/test_model_shapes.py` | starting 3.4 |
| G3: training step | `pytest tests/test_training_step.py` | starting 3.6 |
| G4: SDF properties | `pytest tests/test_sdf_properties.py` | starting 3.7 |
| G5: golden regression | `pytest tests/test_golden.py` | declaring done |

Final acceptance: `pytest tests/ --tb=short` exits 0, and G5 tolerances pass.

---

## Phase 6 — Cleanup

- [ ] Delete `src/tf_compat.py`
- [ ] Remove all `from src.tf_compat import ...` imports
- [ ] Remove TF packages from `requirements.txt` (already done in 2.2)
- [ ] Update `.claude/CLAUDE.md`: change env section to Python 3.12 / PyTorch 2.3
- [ ] Update `README.md` install instructions
- [ ] Commit: `git commit -m "feat: complete PyTorch migration"`

---

## Dependency & Complexity Summary

```
data_layer.py     → copy/minor edit   (no TF deps)
model_utils.py    → low               (cell factories only)
model_base.py     → medium            (session → module, saver → torch.save)
model_GAN.py      → HIGH              (graph → eager, GAN loop, LSTM state)
model_RtnFcst.py  → medium            (supervised, simpler than GAN)
run.py            → low (after model)
run_RtnFcst.py    → low (after model)
create_RF_data.py → copy/minor edit
```

**Recommended order**: data_layer → model_utils → model_base → model_GAN → model_RtnFcst → runners
