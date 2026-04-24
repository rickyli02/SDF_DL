"""Convert TensorFlow RF checkpoints into PyTorch checkpoints.

This script converts the checkpoints in `sample_checkpoints/sample_checkpoints_RF`
into `.pt` files that can be loaded by `src.model.model_RtnFcst.FeedForwardModelWithNA_Return`.

Usage:
    python checkpoint_conversion/convert_rf_checkpoints.py

Requirements:
    - TensorFlow available in the Python interpreter running this script
    - Torch importable either directly or from the known Python 3.11 site-packages
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import ensure_repo_on_path, import_torch

ensure_repo_on_path()
torch = import_torch()

from src.model.model_RtnFcst import FeedForwardModelWithNA_Return


TF_TO_TORCH_KEY_MAP = {
    "Model_Layer/NN_Layer/dense_layer_0/dense/kernel": "_ff.0.weight",
    "Model_Layer/NN_Layer/dense_layer_0/dense/bias": "_ff.0.bias",
    "Model_Layer/NN_Layer/dense_layer_1/dense/kernel": "_ff.1.weight",
    "Model_Layer/NN_Layer/dense_layer_1/dense/bias": "_ff.1.bias",
    "Model_Layer/NN_Layer/dense_layer_2/dense/kernel": "_ff.2.weight",
    "Model_Layer/NN_Layer/dense_layer_2/dense/bias": "_ff.2.bias",
    "Model_Layer/NN_Layer/last_dense_layer/dense/kernel": "_output.weight",
    "Model_Layer/NN_Layer/last_dense_layer/dense/bias": "_output.bias",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_RF/config_RF_1.json",
        help="RF model config used to build the PyTorch module",
    )
    parser.add_argument(
        "--source-root",
        default="sample_checkpoints/sample_checkpoints_RF",
        help="Directory containing RF_1_Trial_* TF checkpoints",
    )
    parser.add_argument(
        "--output-root",
        default="sample_checkpoints/python",
        help="Directory to write converted PyTorch checkpoints",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def convert_rf_checkpoint(checkpoint_prefix: Path, config: dict) -> tuple[dict, dict]:
    reader = tf.train.load_checkpoint(str(checkpoint_prefix))
    model = FeedForwardModelWithNA_Return(config, "test", device="cpu")
    state_dict = model.state_dict()
    mapping = {}

    for tf_name, torch_key in TF_TO_TORCH_KEY_MAP.items():
        value = reader.get_tensor(tf_name)
        if tf_name.endswith("/kernel"):
            tensor = torch.tensor(np.asarray(value).T.copy())
        else:
            tensor = torch.tensor(np.asarray(value).copy())

        expected_shape = tuple(state_dict[torch_key].shape)
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {tf_name} -> {torch_key}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

        state_dict[torch_key] = tensor.to(dtype=state_dict[torch_key].dtype)
        mapping[tf_name] = torch_key

    return state_dict, mapping


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    config = load_config(Path(args.config))

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(Path(args.config)),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "converted_trials": [],
    }

    trial_dirs = sorted(p for p in source_root.iterdir() if p.is_dir() and p.name.startswith("RF_"))
    if not trial_dirs:
        raise FileNotFoundError(f"No RF trial directories found under {source_root}")

    for trial_dir in trial_dirs:
        checkpoint_prefix = trial_dir / "model-best"
        if not (trial_dir / "model-best.index").exists():
            raise FileNotFoundError(f"Missing TF checkpoint index file in {trial_dir}")

        state_dict, mapping = convert_rf_checkpoint(checkpoint_prefix, config)
        output_dir = output_root / trial_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model-000000.pt"

        torch.save(
            {
                "step": 0,
                "state_dict": state_dict,
                "source_checkpoint": str(checkpoint_prefix),
                "mapping": mapping,
            },
            output_path,
        )

        manifest["converted_trials"].append(
            {
                "trial": trial_dir.name,
                "source_checkpoint": str(checkpoint_prefix),
                "output_checkpoint": str(output_path),
            }
        )
        print(f"Converted {trial_dir.name} -> {output_path}")

    manifest_path = output_root / "rf_conversion_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
