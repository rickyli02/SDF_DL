# SDF_DL — Codex Guide

## Project Overview
Deep-learning asset pricing research codebase implementing a GAN-based Stochastic Discount Factor (SDF) network and a return-forecasting ensemble, based on the "Deep Learning in Asset Pricing" paper.

## Repo Layout
```text
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
src/tf_compat.py              # TF1 -> TF2 compat shim (tf.compat.v1)
src/utils.py                  # Misc helpers (deco_print, etc.)
datasets/                     # Runtime data — gitignored
sample_checkpoints/           # Pre-trained checkpoints for reference
```

## Python / TF Environment
- Python 3.11
- TensorFlow 2.15 through `tf.compat.v1`
- Apple Silicon: `tensorflow-macos==2.15.0`
- Virtual environment: `.venv/`

## End-to-End Flow
1. Train SDF:
   `python run.py --config=config/config.json --logdir=output --saveBestFreq=128 --printOnConsole=True --saveLog=True --ignoreEpoch=32`
2. Run the first 8 cells of `model_GAN.ipynb`
3. Build RF data:
   `python create_RF_data.py`
4. Train beta network:
   `python run_RtnFcst_ensembles.py --config config_RF --logdir output_RF --task_id 1 --trial_id 1`
5. Run the remaining notebook cells for EV / XS-R2 results

## Required Dataset Paths
```text
datasets/char/Char_train.npz
datasets/char/Char_valid.npz
datasets/char/Char_test.npz
datasets/macro/macro_train.npz
datasets/macro/macro_valid.npz
datasets/macro/macro_test.npz
```

## Code Notes
- Import TensorFlow via `from src.tf_compat import tf`
- Keep model code in TF1-style graph mode
- `DataInRamInputLayer` expects `.npz` files with `data`, `date`, `variable`, and sometimes `permno`
- Training outputs go to `output/` and `output_RF/`
