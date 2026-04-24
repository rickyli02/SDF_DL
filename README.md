# Deep Learning Asset Pricing — SDF_DL

## Repo Map
- `run.py`: trains the main SDF/GAN model (PyTorch).
- `run_RtnFcst_ensembles.py`: trains the return-forecasting network on the `R * F` data generated from the SDF output.
- `create_RF_data.py`: builds the `datasets/RF/*.npz` files consumed by the return-forecasting stage.
- `model_GAN.ipynb`: analysis notebook used between and after the training stages.
- `capture_golden.py`: saves `.npy` golden regression baselines from a trained PyTorch ensemble.
- `config/config.json`: main SDF training config.
- `config_RF/config_RF_1.json`: return-forecast training config for task 1.
- `src/data/data_layer.py`: loads `.npz` datasets into memory and handles macro feature normalization.
- `src/model/`: PyTorch model definitions and training loops.
- `tf_reference/`: frozen copy of the original TensorFlow source (reference only — not used at runtime).

## Local Setup
Requires Python 3.11+ and PyTorch 2.3+. Apple Silicon (MPS) and CUDA are both supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
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

Scripts fail fast with a clear error if these files are missing.

Datasets: [Google Drive](https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l?usp=sharing)

## End-to-End Flow

### Step 1: Train the SDF network
Runs 9 trials (Task_1_Trial_0 … Task_1_Trial_8) sequentially and saves `.pt` checkpoints under `output/`.

```bash
python run.py --config config/config.json --logdir output
```

Optional flags: `--seed 42`, `--num-trials 9`, `--saveBestFreq 128`, `--printOnConsole`.

### Step 2: Analyse SDF outputs (notebook stage 1)
Open `model_GAN.ipynb` and run through the **Model Performance** section (cells up to and including the Sharpe print). This produces `output/sdf_normalized_ensemble.npy`.

### Step 3: Generate the `R * F` datasets
```bash
python create_RF_data.py
```

### Step 4: Train the beta prediction network
```bash
python run_RtnFcst_ensembles.py --config config_RF --logdir output_RF --task_id 1 --trial_id 1
```

### Step 5: Analyse predictive performance (notebook stage 2)
Run the remaining cells of `model_GAN.ipynb` to compute EV and XS-R² pricing results.

## Regression Tests
```bash
pytest tests/ -v
```

Golden-value regression tests (`tests/test_golden.py`) require trained checkpoints and pre-saved baselines. To create baselines after training:

```bash
python capture_golden.py --logdir output
pytest tests/test_golden.py -v
```
