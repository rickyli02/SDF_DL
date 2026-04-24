"""Golden-value regression tests for the RF / beta model."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.data.data_layer import DataInRamInputLayer
from src.model.model_RtnFcst import FeedForwardModelWithNA_Return_Ensembled


GOLDEN_DIR = ".claude/golden"
CONFIG_PATH = "config_RF/config_RF_1.json"
CHECKPOINT_BASE = "sample_checkpoints/python"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "rf_prediction_test.npy")),
    reason="RF golden files not found — run `python capture_golden_rf.py` first",
)


def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def golden():
    return {
        "pred": np.load(os.path.join(GOLDEN_DIR, "rf_prediction_test.npy")),
        "factor": np.load(os.path.join(GOLDEN_DIR, "rf_factor_test.npy")),
        "sharpe": float(np.load(os.path.join(GOLDEN_DIR, "rf_sharpe_test.npy"))),
    }


@pytest.fixture(scope="module")
def pytorch_outputs():
    with open(CONFIG_PATH) as fh:
        config = json.load(fh)

    logdirs = [
        os.path.join(CHECKPOINT_BASE, f"RF_1_Trial_{i}")
        for i in range(9)
        if os.path.isdir(os.path.join(CHECKPOINT_BASE, f"RF_1_Trial_{i}"))
    ]
    if not logdirs:
        pytest.skip("Converted RF checkpoints not found under sample_checkpoints/python")

    dl_test = DataInRamInputLayer(
        config["individual_feature_file_test"],
        pathMacroFeature=config["macro_feature_file_test"],
        macroIdx=config["macro_idx"],
    )
    model = FeedForwardModelWithNA_Return_Ensembled(logdirs, config, "test", device="cpu")

    pred = model.getPrediction(dl_test)
    factor = model.getSDFFactor(dl_test)
    sharpe = model.evaluate_sharpe(dl_test)
    return {"pred": pred, "factor": factor, "sharpe": sharpe}


def test_rf_prediction_correlation(pytorch_outputs, golden):
    corr = pearson(pytorch_outputs["pred"], golden["pred"])
    assert corr >= 0.999999, f"RF prediction correlation {corr:.6f} < 0.999999"


def test_rf_factor_correlation(pytorch_outputs, golden):
    corr = pearson(pytorch_outputs["factor"], golden["factor"])
    assert corr >= 0.999999, f"RF factor correlation {corr:.6f} < 0.999999"


def test_rf_sharpe_close(pytorch_outputs, golden):
    diff = abs(float(pytorch_outputs["sharpe"]) - golden["sharpe"])
    assert diff <= 1e-8, f"RF sharpe drift {diff:.3e} > 1e-8"

