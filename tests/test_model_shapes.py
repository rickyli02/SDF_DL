"""
Tests for model output shapes.

Verifies that getWeightWithData and getSDF return arrays with the expected
dimensions regardless of the underlying framework.

TF note: these tests were written for the TF1 (tf.compat.v1) codebase.
They are skipped in the current environment because TF 2.20 (Keras 3) removed
BasicRNNCell from tf.compat.v1.nn.rnn_cell — the TF code can no longer run.
After the PyTorch migration: remove the skip, replace session boilerplate with
`model.forward(...)`, keep all assertions unchanged.
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
def sdf_model_and_data(synthetic_char, synthetic_macro, minimal_config):
    config = minimal_config
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)

    global_step = tf.train.get_or_create_global_step()
    model = FeedForwardModelWithNA_GAN(config, "test", config["tSize_test"], global_step=global_step)

    sess = tf.Session()
    model.randomInitialization(sess)
    yield model, dl, sess
    sess.close()


def test_get_weight_shape(sdf_model_and_data, minimal_config):
    model, dl, sess = sdf_model_and_data
    weights = model.getWeightWithData(sess, dl)
    # weights should have one entry per valid (firm, date) pair
    for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
        expected_len = mask.sum()
    assert weights.shape == (expected_len,)
    assert np.isfinite(weights).all()


def test_get_sdf_shape(sdf_model_and_data):
    model, dl, sess = sdf_model_and_data
    sdf = model.getSDF(sess, dl)
    from tests.conftest import T
    assert sdf.shape == (T, 1)
    assert np.isfinite(sdf).all()


def test_get_sdf_factor_shape(sdf_model_and_data):
    model, dl, sess = sdf_model_and_data
    factor = model.getSDFFactor(sess, dl)
    from tests.conftest import T
    assert factor.shape == (T, 1)


def test_normalized_sdf_shape(sdf_model_and_data):
    model, dl, sess = sdf_model_and_data
    sdf_norm = model.getNormalizedSDF(sess, dl)
    from tests.conftest import T
    assert sdf_norm.shape == (T, 1)