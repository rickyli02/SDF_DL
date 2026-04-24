"""
Golden-value regression tests.

Compares a live PyTorch model inference against .npy baseline files created by
capture_golden.py. These tests act as a numerical regression gate: once you have
a trained model, run capture_golden.py to fix the baseline, then future refactors
must not move the outputs beyond the tolerances below.

Workflow:
  1. Train:   python run.py --config config/config.json --logdir output
  2. Capture: python capture_golden.py --logdir output
  3. Test:    pytest tests/test_golden.py

Tolerances:
  - SDF values:        Pearson correlation ≥ 0.99
  - Weight vector:     Pearson correlation ≥ 0.99
  - Sharpe (scalar):   absolute difference ≤ 0.05
  - Normalized SDF:    Pearson correlation ≥ 0.99
"""

import glob
import json
import os

import numpy as np
import pytest

from src.data.data_layer import DataInRamInputLayer
from src.model.model_GAN import FeedForwardModelWithNA_GAN_Ensembled

GOLDEN_DIR = ".claude/golden"
CONFIG_PATH = "config/config.json"

# ---------------------------------------------------------------------------
# Module-level skip: golden files must exist (created by capture_golden.py)
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "sdf_test.npy")),
    reason="Golden files not found — run `python capture_golden.py --logdir <trained_logdir>` first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def annualised_sharpe(sdf):
    returns = sdf.ravel() - 1
    return float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(12))


def _find_logdirs(base, num_trials=9):
    """Return existing Task_1_Trial_*/sharpe dirs under base."""
    dirs = []
    for i in range(num_trials):
        d = os.path.join(base, f"Task_1_Trial_{i}", "sharpe")
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def _load_test_data():
    if not os.path.exists(CONFIG_PATH):
        return None, None
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    if "macro_idx" not in config:
        config["macro_idx"] = None
    char_test = config.get("individual_feature_file_test", "")
    if not os.path.exists(char_test):
        return None, None
    dl_train = DataInRamInputLayer(
        config["individual_feature_file"],
        pathMacroFeature=config["macro_feature_file"],
        macroIdx=config["macro_idx"],
    )
    mean_macro, std_macro = dl_train.getMacroFeatureMeanStd()
    dl_test = DataInRamInputLayer(
        char_test,
        pathMacroFeature=config["macro_feature_file_test"],
        macroIdx=config["macro_idx"],
        meanMacroFeature=mean_macro,
        stdMacroFeature=std_macro,
    )
    return config, dl_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden():
    return {
        "sdf":      np.load(os.path.join(GOLDEN_DIR, "sdf_test.npy")),
        "weights":  np.load(os.path.join(GOLDEN_DIR, "weights_test.npy")),
        "sdf_norm": np.load(os.path.join(GOLDEN_DIR, "sdf_normalized_ensemble.npy")),
    }


@pytest.fixture(scope="module")
def pytorch_outputs():
    """Run the PyTorch ensemble on the test split and return inference outputs.

    Skips if:
    - Real datasets are missing (gitignored)
    - No trained PyTorch checkpoints are found

    To enable: train a model, then run capture_golden.py (which also records
    the logdir used). The fixture auto-discovers trial dirs under 'output/'.
    """
    config, dl_test = _load_test_data()
    if config is None or dl_test is None:
        pytest.skip("Real test dataset not available — golden tests require full datasets")

    # Look for trained checkpoints; prefer output/ then sample_checkpoints/
    candidate_bases = ["output", "sample_checkpoints/sample_checkpoints"]
    logdirs = []
    for base in candidate_bases:
        logdirs = _find_logdirs(base)
        if logdirs:
            break

    if not logdirs:
        pytest.skip(
            "No trained PyTorch checkpoints found under output/ — "
            "run `python run.py --config config/config.json --logdir output` first"
        )

    model = FeedForwardModelWithNA_GAN_Ensembled(logdirs, config, "test")
    sdf = model.getSDF(dl_test)
    weights = model.getWeightWithData(dl_test)
    sdf_norm = model.getNormalizedSDF(dl_test)
    return {"sdf": sdf, "weights": weights, "sdf_norm": sdf_norm}


# ---------------------------------------------------------------------------
# Assertions — these are the acceptance criteria; do not change tolerances
# ---------------------------------------------------------------------------

def test_sdf_correlation(pytorch_outputs, golden):
    corr = pearson(pytorch_outputs["sdf"], golden["sdf"])
    assert corr >= 0.99, f"SDF Pearson correlation {corr:.4f} < 0.99"


def test_weight_correlation(pytorch_outputs, golden):
    corr = pearson(pytorch_outputs["weights"], golden["weights"])
    assert corr >= 0.99, f"Weight Pearson correlation {corr:.4f} < 0.99"


def test_sharpe_close(pytorch_outputs, golden):
    sr_pt = annualised_sharpe(pytorch_outputs["sdf"])
    sr_base = annualised_sharpe(golden["sdf"])
    assert abs(sr_pt - sr_base) <= 0.05, (
        f"Sharpe divergence: current={sr_pt:.3f}, baseline={sr_base:.3f}, "
        f"diff={abs(sr_pt - sr_base):.3f}"
    )


def test_normalized_sdf_correlation(pytorch_outputs, golden):
    corr = pearson(pytorch_outputs["sdf_norm"], golden["sdf_norm"])
    assert corr >= 0.99, f"Normalized SDF correlation {corr:.4f} < 0.99"