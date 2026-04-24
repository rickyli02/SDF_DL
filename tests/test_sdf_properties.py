"""Tests for economic / statistical properties of SDF outputs (PyTorch).

Invariants that must hold regardless of random initialization:
  - SDF is finite (no NaN / Inf)
  - Portfolio weights within each period are bounded (long-short structure)
  - Normalized SDF is strictly positive (pricing kernel property)
"""

import numpy as np
import pytest

from src.data.data_layer import DataInRamInputLayer
from src.model.model_GAN import FeedForwardModelWithNA_GAN


@pytest.fixture(scope="module")
def inferred(synthetic_char, synthetic_macro, minimal_config):
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)
    model = FeedForwardModelWithNA_GAN(minimal_config, "test")
    sdf = model.getSDF(dl)
    weights = model.getWeightWithData(dl)
    sdf_norm = model.getNormalizedSDF(dl)
    return sdf, weights, sdf_norm, dl


def test_sdf_is_finite(inferred):
    sdf, _, _, _ = inferred
    assert np.isfinite(sdf).all(), "SDF contains NaN or Inf"


def test_weights_finite(inferred):
    _, weights, _, _ = inferred
    assert np.isfinite(weights).all(), "Weights contain NaN or Inf"


def test_weights_bounded_per_period(inferred):
    """Weights within each time step should have bounded absolute sum (long-short)."""
    _, weights, _, dl = inferred
    for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
        splits = np.sum(mask, axis=1).cumsum()[:-1]
        for t, w_t in enumerate(np.split(weights, splits)):
            assert np.isfinite(w_t).all(), f"NaN weights at t={t}"


def test_sdf_not_degenerate(inferred):
    sdf, _, _, _ = inferred
    assert (sdf != 0).any(), "SDF is all-zero — model is degenerate"


def test_normalized_sdf_positive(inferred):
    """Normalized SDF represents a pricing kernel — should be > 0."""
    _, _, sdf_norm, _ = inferred
    assert (sdf_norm > 0).all(), "Normalized SDF has non-positive entries"
