# SDF_DL — Claude Code Guide

## Project Overview
Deep-learning asset pricing research codebase implementing a GAN-based Stochastic Discount Factor (SDF) network and a return-forecasting ensemble, based on the "Deep Learning in Asset Pricing" paper.

## Repo Layout
```
run.py                        # Stage 1: train the SDF/GAN model
run_RtnFcst_ensembles.py      # Stage 4: train return-forecasting beta network
create_RF_data.py             # Stage 3: build datasets/RF/*.npz from SDF output
capture_golden.py             # Save .npy golden regression baselines
model_GAN.ipynb               # Analysis notebook (stages 2 & 5)
config/config.json            # SDF training hyperparameters
config_RF/config_RF_1.json    # Return-forecast config (task 1)
src/data/data_layer.py        # Loads .npz datasets; macro feature normalization
src/model/model_GAN.py        # GAN/SDF model definition + training loop (PyTorch)
src/model/model_RtnFcst.py    # Return-forecasting model (PyTorch)
src/model/model_base.py       # Shared nn.Module base class
src/model/model_utils.py      # Shared utilities (getFactor, calculateStatistics, etc.)
src/utils.py                  # Misc helpers (deco_print, sharpe, etc.)
tf_reference/                 # Frozen original TensorFlow source (reference only)
datasets/                     # Runtime data — gitignored
sample_checkpoints/           # Pre-trained TF checkpoints (historical reference only)
```

## Python / PyTorch Environment
- Python 3.11+, PyTorch 2.3+ (`torch`, `tensorboard`)
- Apple Silicon: MPS device used automatically when available
- Venv at `.venv/` — activate with `source .venv/bin/activate`
- No TensorFlow dependency — all model code is pure PyTorch (`nn.Module`)

## End-to-End Training Flow
1. **Train SDF** — `python run.py --config config/config.json --logdir output`
2. **Notebook stage 1** — run SDF analysis cells of `model_GAN.ipynb` to produce `output/sdf_normalized_ensemble.npy`
3. **Build RF data** — `python create_RF_data.py`
4. **Train beta network** — `python run_RtnFcst_ensembles.py --config config_RF --logdir output_RF --task_id 1 --trial_id 1`
5. **Notebook stage 2** — run remaining cells of `model_GAN.ipynb` for EV / XS-R² results

## Required Dataset Paths (gitignored)
```
datasets/char/Char_{train,valid,test}.npz
datasets/macro/macro_{train,valid,test}.npz
```
Scripts fail fast with a clear message if these are missing. Download from the Google Drive link in README.md.

## Key Config Fields (`config/config.json`)
| Field | Meaning |
|---|---|
| `hidden_dim` | FF layer widths for SDF network |
| `num_units_rnn` | LSTM hidden size |
| `tSize / tSize_valid / tSize_test` | Time-series window sizes |
| `individual_feature_dim` | Number of firm characteristics (46) |
| `macro_feature_dim` | Number of macro features (178) |
| `num_epochs` | Total training epochs |
| `dropout` | Dropout keep probability (0.95 → maps to `p=0.05` in PyTorch) |

## Coding Conventions
- All model code uses `torch.nn.Module` — no TF/session APIs anywhere
- `.npz` files accessed via `data_layer.DataInRamInputLayer`
- Checkpoints saved as `model-{step:06d}.pt` under `logdir/`; loaded via `model.load(logdir)`
- Outputs go to `output/` or `output_RF/` (both gitignored)
- Tests live in `tests/`; run with `pytest tests/ -v`