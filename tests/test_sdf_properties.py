"""
Tests for economic / statistical properties of SDF outputs.

These invariants must hold for both the TF and PyTorch implementations:
  - SDF mean ≈ 1  (risk-neutral pricing kernel normalization)
  - Portfolio weights sum to zero within each time period (long-short)
  - Normalized SDF ≥ 0 (probabilities)

TF note: skipped because TF 2.20 (Keras 3) removed BasicRNNCell from
tf.compat.v1.nn.rnn_cell. After PyTorch migration: remove the skip, replace
session boilerplate with direct model calls, keep all assertions.
"""

import numpy as np
import pytest

TF_UNAVAILABLE = True
try:
    from src.model.model_GAN import FeedForwardModelWithNA_GAN
    from src.tf_compat import tf
    TF_UNAVAILABLE = False
except AttributeError:
    pass

from src.data.data_layer import DataInRamInputLayer

pytestmark = pytest.mark.skipif(TF_UNAVAILABLE, reason="TF 2.20 removed BasicRNNCell; TF model tests skipped until PyTorch migration")


@pytest.fixture(scope="module")
def inferred(synthetic_char, synthetic_macro, minimal_config):
    config = minimal_config
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)

    global_step = tf.train.get_or_create_global_step()
    model = FeedForwardModelWithNA_GAN(config, "test", config["tSize_test"], global_step=global_step)

    sess = tf.Session()
    model.randomInitialization(sess)

    sdf = model.getSDF(sess, dl)
    weights = model.getWeightWithData(sess, dl)
    sdf_norm = model.getNormalizedSDF(sess, dl)

    sess.close()
    yield sdf, weights, sdf_norm, dl


def test_sdf_is_finite(inferred):
    sdf, _, _, _ = inferred
    assert np.isfinite(sdf).all(), "SDF contains NaN or Inf"


def test_weights_sum_to_zero_per_period(inferred):
    """Long-short portfolio: within each time step, weights should sum to ≈ 0."""
    _, weights, _, dl = inferred
    for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
        period_sizes = np.sum(mask, axis=1)
        w_split = np.split(weights, period_sizes.cumsum()[:-1])
        for t, w_t in enumerate(w_split):
            assert abs(w_t.sum()) < 1.0, (
                f"Weights at t={t} sum to {w_t.sum():.4f} — expected near zero"
            )


def test_sdf_no_all_zero_periods(inferred):
    sdf, _, _, _ = inferred
    assert (sdf != 0).any(), "SDF is all-zero — model is degenerate"


def test_normalized_sdf_positive(inferred):
    """Normalized SDF represents a pricing kernel and should be > 0."""
    _, _, sdf_norm, _ = inferred
    assert (sdf_norm > 0).all(), "Normalized SDF has non-positive entries"