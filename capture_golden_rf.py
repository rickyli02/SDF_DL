"""Capture RF golden baselines from converted PyTorch RF checkpoints.

Usage:
    python capture_golden_rf.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.data.data_layer import DataInRamInputLayer
from src.model.model_RtnFcst import FeedForwardModelWithNA_Return_Ensembled


GOLDEN_DIR = ".claude/golden"
CONFIG_PATH = "config_RF/config_RF_1.json"
CHECKPOINT_BASE = "sample_checkpoints/python"
NUM_TRIALS = 9


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    with open(CONFIG_PATH) as fh:
        config = json.load(fh)

    logdirs = [
        os.path.join(CHECKPOINT_BASE, f"RF_1_Trial_{i}")
        for i in range(NUM_TRIALS)
        if os.path.isdir(os.path.join(CHECKPOINT_BASE, f"RF_1_Trial_{i}"))
    ]
    if not logdirs:
        raise FileNotFoundError(
            f"No converted RF checkpoints found under {CHECKPOINT_BASE}. "
            "Run checkpoint_conversion/convert_rf_checkpoints.py first."
        )

    dl_test = DataInRamInputLayer(
        config["individual_feature_file_test"],
        pathMacroFeature=config["macro_feature_file_test"],
        macroIdx=config["macro_idx"],
    )
    model = FeedForwardModelWithNA_Return_Ensembled(logdirs, config, "test", device="cpu")

    pred = model.getPrediction(dl_test)
    factor = model.getSDFFactor(dl_test)
    sharpe = np.array(model.evaluate_sharpe(dl_test), dtype=np.float64)

    np.save(os.path.join(GOLDEN_DIR, "rf_prediction_test.npy"), pred)
    np.save(os.path.join(GOLDEN_DIR, "rf_factor_test.npy"), factor)
    np.save(os.path.join(GOLDEN_DIR, "rf_sharpe_test.npy"), sharpe)

    print(f"Saved RF golden files to {GOLDEN_DIR}/")
    print(f"  rf_prediction_test.npy  shape={pred.shape}")
    print(f"  rf_factor_test.npy      shape={factor.shape}")
    print(f"  rf_sharpe_test.npy      value={float(sharpe):.6f}")


if __name__ == "__main__":
    main()

