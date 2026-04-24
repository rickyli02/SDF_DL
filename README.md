# Deep Learning Asset Pricing Main Code for Training

## Repo Map
- `run.py`: trains the main SDF/GAN model.
- `run_RtnFcst_ensembles.py`: trains the return-forecasting network on the `R * F` data generated from the SDF output.
- `create_RF_data.py`: builds the `datasets/RF/*.npz` files consumed by the return-forecasting stage.
- `model_GAN.ipynb`: analysis notebook used between and after the training stages.
- `config/config.json`: main SDF training config.
- `config_RF/config_RF_1.json`: return-forecast training config for task 1.
- `src/data/data_layer.py`: loads `.npz` datasets into memory and handles macro feature normalization.
- `src/model/`: model definitions and training loops.

## Runtime Notes
- The original codebase targeted TensorFlow 1.12 / Python 3.6.
- This repo has been adjusted to run in TensorFlow 2.15 compatibility mode (`tf.compat.v1`) so it can be used on a current Python 3.11 environment.
- The scripts still expect the original dataset layout from the paper codebase.

## Local Setup
These steps are intended for this repo on macOS Apple Silicon and also work with standard `tensorflow==2.15.x` on non-Apple-Silicon machines.

```bash
/usr/local/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Layout
Download the dataset bundle from the project Google Drive and place it under the repo root so these paths exist:

```text
datasets/char/Char_train.npz
datasets/char/Char_valid.npz
datasets/char/Char_test.npz
datasets/macro/macro_train.npz
datasets/macro/macro_valid.npz
datasets/macro/macro_test.npz
```

The scripts now fail fast with a clear error if these files are missing.

Datasets: [Google Drive](https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l?usp=sharing)

## End-to-End Flow
### Step 1: Train the SDF network
```bash
python run.py --config=config/config.json --logdir=output --saveBestFreq=128 --printOnConsole=True --saveLog=True --ignoreEpoch=32
```

### Step 2: Run the first 8 cells of `model_GAN.ipynb`
This produces the SDF outputs used in the next stage.

### Step 3: Generate the `R * F` datasets
```bash
python create_RF_data.py
```

### Step 4: Train the beta prediction network
```bash
python run_RtnFcst_ensembles.py --config config_RF --logdir output_RF --task_id 1 --trial_id 1
```

### Step 5: Run the remaining cells of `model_GAN.ipynb`
This computes the EV and XS-R2 pricing results.
