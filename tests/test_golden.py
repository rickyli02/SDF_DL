"""
Golden-value regression tests.

Compares PyTorch model outputs against the .npy files saved by capture_golden.py
from the pre-trained TF checkpoints.

Run capture_golden.py once against the TF codebase before running these tests.

Tolerances:
  - SDF values:    Pearson correlation ≥ 0.99
  - Weight vector: Pearson correlation ≥ 0.99
  - Sharpe (scalar): absolute difference ≤ 0.05
"""

import os
import numpy as np
import pytest

GOLDEN_DIR = ".claude/golden"

# Skip the whole module if golden files haven't been captured yet
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "sdf_test.npy")),
    reason="Golden files not found — run capture_golden.py first",
)


@pytest.fixture(scope="module")
def golden():
    return {
        "sdf": np.load(os.path.join(GOLDEN_DIR, "sdf_test.npy")),
        "weights": np.load(os.path.join(GOLDEN_DIR, "weights_test.npy")),
        "sdf_norm": np.load(os.path.join(GOLDEN_DIR, "sdf_normalized_ensemble.npy")),
    }


def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    return np.corrcoef(a, b)[0, 1]


def sharpe(sdf):
    returns = sdf.ravel() - 1
    return returns.mean() / (returns.std() + 1e-8) * np.sqrt(12)


# ---------------------------------------------------------------------------
# The fixtures below must be implemented by the NEW PyTorch code.
# Fill in the import and construction once the PyTorch model exists.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pytorch_outputs():
    """
    Return a dict with keys: sdf, weights, sdf_norm.
    Replace this stub with the actual PyTorch model inference.
    """
    pytest.skip("PyTorch model not yet implemented — stub fixture")


# ---------------------------------------------------------------------------
# Assertions (do not change these — they are the acceptance criteria)
# ---------------------------------------------------------------------------

def test_sdf_correlation(pytorch_outputs, golden):
    corr = pearson(pytorch_outputs["sdf"], golden["sdf"])
    assert corr >= 0.99, f"SDF Pearson correlation {corr:.4f} < 0.99"


def test_weight_correlation(pytorch_outputs, golden):
    corr = pearson(pytorch_outputs["weights"], golden["weights"])
    assert corr >= 0.99, f"Weight Pearson correlation {corr:.4f} < 0.99"


def test_sharpe_close(pytorch_outputs, golden):
    sr_pt = sharpe(pytorch_outputs["sdf"])
    sr_tf = sharpe(golden["sdf"])
    assert abs(sr_pt - sr_tf) <= 0.05, (
        f"Sharpe divergence: PyTorch={sr_pt:.3f}, TF={sr_tf:.3f}, diff={abs(sr_pt-sr_tf):.3f}"
    )


def test_normalized_sdf_correlation(pytorch_outputs, golden):
    corr = pearson(pytorch_outputs["sdf_norm"], golden["sdf_norm"])
    assert corr >= 0.99, f"Normalized SDF correlation {corr:.4f} < 0.99"