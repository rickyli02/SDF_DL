"""
Golden-value capture script (PyTorch).

Loads a trained SDF ensemble from a logdir, runs inference on the test split,
and saves outputs to .claude/golden/ as .npy files. These files become the
numerical regression baseline for test_golden.py.

Workflow:
  1. Train the SDF model:  python run.py --config config/config.json --logdir output
  2. Run this script:       python capture_golden.py --logdir output
  3. Run golden tests:      pytest tests/test_golden.py

Usage:
    python capture_golden.py [--logdir output] [--config config/config.json]
                             [--num-trials 9] [--golden-dir .claude/golden]

The script expects logdir to contain sub-directories named Task_1_Trial_*/sharpe
(matching the output layout from run.py). Pass --num-trials to control how many
trial directories to aggregate.

If --logdir is not given, it defaults to 'output'.
"""

import argparse
import json
import os
import numpy as np

from src.data import data_layer
from src.model.model_GAN import FeedForwardModelWithNA_GAN_Ensembled


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="output",
                   help="Directory containing Task_1_Trial_*/sharpe checkpoints")
    p.add_argument("--config", default="config/config.json")
    p.add_argument("--num-trials", type=int, default=9)
    p.add_argument("--golden-dir", default=".claude/golden")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.golden_dir, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)
    if "macro_idx" not in config:
        config["macro_idx"] = None

    # Build macro normalization stats from the training split
    print("Loading training data for macro normalization stats...")
    dl_train = data_layer.DataInRamInputLayer(
        config["individual_feature_file"],
        pathMacroFeature=config["macro_feature_file"],
        macroIdx=config["macro_idx"],
    )
    mean_macro, std_macro = dl_train.getMacroFeatureMeanStd()

    print("Loading test data...")
    dl_test = data_layer.DataInRamInputLayer(
        config["individual_feature_file_test"],
        pathMacroFeature=config["macro_feature_file_test"],
        macroIdx=config["macro_idx"],
        meanMacroFeature=mean_macro,
        stdMacroFeature=std_macro,
    )

    # Collect checkpoint dirs; skip any that don't exist
    logdirs = []
    for i in range(args.num_trials):
        d = os.path.join(args.logdir, f"Task_1_Trial_{i}", "sharpe")
        if os.path.isdir(d):
            logdirs.append(d)
    if not logdirs:
        raise FileNotFoundError(
            f"No Task_1_Trial_*/sharpe directories found under {args.logdir}.\n"
            "Train the model first with: python run.py --config config/config.json --logdir output"
        )
    print(f"Found {len(logdirs)} trial checkpoint(s) in {args.logdir}")

    model = FeedForwardModelWithNA_GAN_Ensembled(logdirs, config, "test")

    print("Capturing SDF on test split...")
    sdf = model.getSDF(dl_test)
    out = os.path.join(args.golden_dir, "sdf_test.npy")
    np.save(out, sdf)
    print(f"  {out}  shape={sdf.shape}  mean={sdf.mean():.6f}  std={sdf.std():.6f}")

    print("Capturing SDF weights on test split...")
    weights = model.getWeightWithData(dl_test)
    out = os.path.join(args.golden_dir, "weights_test.npy")
    np.save(out, weights)
    print(f"  {out}  shape={weights.shape}  mean={weights.mean():.6f}")

    print("Capturing normalized SDF on test split...")
    sdf_norm = model.getNormalizedSDF(dl_test)
    out = os.path.join(args.golden_dir, "sdf_normalized_ensemble.npy")
    np.save(out, sdf_norm)
    print(f"  {out}  shape={sdf_norm.shape}  mean={sdf_norm.mean():.6f}")

    print(f"\nAll golden files saved to {args.golden_dir}/")


if __name__ == "__main__":
    main()