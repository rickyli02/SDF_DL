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
Two workflows are supported:

#### Option A: Single run
Runs one PyTorch SDF training job and saves checkpoints/logs under `output/`.

```bash
python run.py --config config/config.json --logdir output
```

Useful flags: `--seed 42`, `--saveBestFreq 128`, `--printOnConsole true`, `--saveLog true`, `--printFreq 128`, `--ignoreEpoch 64`.

This produces outputs such as:

```text
output/sharpe/model-000249.pt
output/loss/model-000318.pt
output/model-001342.pt
```

#### Option B: Manual multi-trial layout
If you want the old-style `Task_1_Trial_*` structure used by earlier notebook code and ensemble tooling, run multiple jobs manually with separate logdirs:

```bash
for i in {0..8}; do
  python run.py \
    --config config/config.json \
    --logdir output/Task_1_Trial_$i \
    --seed $i
done
```

This produces per-trial folders such as:

```text
output/Task_1_Trial_0/sharpe/
output/Task_1_Trial_1/sharpe/
...
output/Task_1_Trial_8/sharpe/
```

### Step 2: Analyse SDF outputs (notebook stage 1)
Open `model_GAN.ipynb` and run through the **Model Performance** section (cells up to and including the Sharpe print). This produces `output/sdf_normalized_ensemble.npy`.

Important:

- The notebook path settings must match the training layout you used.
- If you used the single-run layout, notebook cells that currently refer to `output/Task_1_Trial_*` need to be changed to point at the single-run paths, such as `output/sharpe`.
- If you used the manual multi-trial layout, notebook cells can keep using the `output/Task_1_Trial_*` structure.

### Step 3: Generate the `R * F` datasets
```bash
python create_RF_data.py
```

This script is silent on success. After it runs, the RF datasets should exist at:

```text
datasets/RF/RF_train_normalized_task_1.npz
datasets/RF/RF_valid_normalized_task_1.npz
datasets/RF/RF_test_normalized_task_1.npz
```

### Step 4: Train the beta prediction network
```bash
python run_RtnFcst_ensembles.py --config config_RF --logdir output_RF --task_id 1 --trial_id 1
```

This writes trial outputs under:

```text
output_RF/RF_1_Trial_1/
```

including TensorBoard event files and saved `.pt` checkpoints such as `model-000000.pt`, `model-000001.pt`, etc.

If you want a multi-trial RF layout similar to the older workflow, run multiple trial ids manually:

```bash
for i in {0..8}; do
  python run_RtnFcst_ensembles.py \
    --config config_RF \
    --logdir output_RF \
    --task_id 1 \
    --trial_id $i
done
```

This produces:

```text
output_RF/RF_1_Trial_0/
output_RF/RF_1_Trial_1/
...
output_RF/RF_1_Trial_8/
```

### Step 5: Analyse predictive performance (notebook stage 2)
Run the remaining cells of `model_GAN.ipynb` to compute EV and XS-R² pricing results.

Again, the corresponding notebook paths must be updated to match whichever RF output layout you actually used.

## Regression Tests
```bash
pytest tests/ -v
```

Golden-value regression tests (`tests/test_golden.py`) require trained checkpoints and pre-saved baselines. To create baselines after training:

```bash
python capture_golden.py --logdir output
pytest tests/test_golden.py -v
```

## Current Observed Result
The following result was obtained from a local Apple Silicon / MPS training run using:

```bash
python run.py \
  --config config/config.json \
  --logdir output \
  --saveBestFreq 128 \
  --printOnConsole true \
  --saveLog true \
  --printFreq 128 \
  --ignoreEpoch 64
```

Final summary from that run:

- Best validation epoch: `249`
- Best Sharpe at that epoch: `Train 0.973  Valid 0.333  Test 0.255`
- Corresponding checkpoint path: `output/sharpe/model-000249.pt`

Interpretation:

- The unconditional phase produced the best validation Sharpe in this run.
- The later conditional / GAN phase improved training Sharpe further, but did not beat the best validation Sharpe reached earlier.
- This means the current training pipeline is working end to end, but the adversarial phase appears less stable than the unconditional phase on this configuration.

### RF / Beta Model Result
After generating RF datasets with:

```bash
python create_RF_data.py
```

the beta network was trained with:

```bash
python run_RtnFcst_ensembles.py --config config_RF --logdir output_RF --task_id 1 --trial_id 1
```

Observed result from that run:

- Best validation epoch (by reported Sharpe): `842`
- Best Sharpe at that epoch: `Train 0.892  Valid 0.809  Test 0.423`
- Output directory: `output_RF/RF_1_Trial_1/`

Additional notes:

- The RF datasets were created successfully and are present under `datasets/RF/`.
- The RF training run completed end to end on `mps`.
- The checkpoint directory contains multiple `.pt` snapshots plus TensorBoard event files.
- The printed "best valid epoch" is based on the tracked validation Sharpe; checkpoint saving in the current RF training loop is still driven by validation loss.
