# SDF_DL — Claude Code Guide

## Project Overview
Deep-learning asset pricing research codebase implementing a GAN-based Stochastic Discount Factor (SDF) network and a return-forecasting ensemble, based on the "Deep Learning in Asset Pricing" paper.

## Repo Layout
```
run.py                        # Stage 1: train the SDF/GAN model
run_RtnFcst_ensembles.py      # Stage 4: train return-forecasting beta network
create_RF_data.py             # Stage 3: build datasets/RF/*.npz from SDF output
model_GAN.ipynb               # Analysis notebook (stages 2 & 5)
config/config.json            # SDF training hyperparameters
config_RF/config_RF_1.json    # Return-forecast config (task 1)
src/data/data_layer.py        # Loads .npz datasets; macro feature normalization
src/model/model_GAN.py        # GAN/SDF model definition + training loop
src/model/model_RtnFcst.py    # Return-forecasting model
src/model/model_base.py       # Shared base model
src/model/model_utils.py      # Shared utilities
src/tf_compat.py              # TF1 → TF2 compat shim (tf.compat.v1)
src/utils.py                  # Misc helpers (deco_print, etc.)
datasets/                     # Runtime data — gitignored
sample_checkpoints/           # Pre-trained checkpoints for reference
```

## Python / TF Environment
- Python 3.11, TensorFlow 2.15 (Apple Silicon: `tensorflow-macos==2.15.0`)
- All TF1-style code goes through `src/tf_compat.py` (`tf.compat.v1`)
- Venv at `.venv/` — activate with `source .venv/bin/activate`

## End-to-End Training Flow
1. **Train SDF** — `python run.py --config=config/config.json --logdir=output --saveBestFreq=128 --printOnConsole=True --saveLog=True --ignoreEpoch=32`
2. **Notebook stage 1** — run first 8 cells of `model_GAN.ipynb` to produce SDF outputs
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
| `dropout` | Keep probability (0.95) |

## Coding Conventions
- Use `tf.compat.v1` exclusively — do not call TF2 eager APIs in model files
- Import TF via `from src.tf_compat import tf`
- `.npz` files accessed via `data_layer.DataInRamInputLayer`
- Outputs go to `output/` or `output_RF/` (both gitignored)
