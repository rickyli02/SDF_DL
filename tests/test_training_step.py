"""
Tests that a single training step completes without NaN losses.

TF note: skipped because TF 2.20 (Keras 3) removed BasicRNNCell from
tf.compat.v1.nn.rnn_cell. After PyTorch migration: remove the skip, replace
session boilerplate with the new training loop interface, keep all assertions.
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
def train_setup(synthetic_char, synthetic_macro, minimal_config):
    config = minimal_config
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)

    global_step = tf.train.get_or_create_global_step()
    model = FeedForwardModelWithNA_GAN(config, "train", config["tSize"], global_step=global_step)

    sess = tf.Session()
    model.randomInitialization(sess)
    yield model, dl, sess, config
    sess.close()


def test_one_training_step_no_nan(train_setup, tmp_path):
    model, dl, sess, config = train_setup
    logdir = str(tmp_path / "ckpt")

    # run a single sub-epoch training pass and check Sharpe is a finite number
    sharpe_train, sharpe_valid, sharpe_test = model.train(
        sess, dl, dl, logdir,
        model_valid=FeedForwardModelWithNA_GAN(config, "valid", config["tSize_valid"], force_var_reuse=True),
        loss_weight=None, loss_weight_valid=None,
        dl_test=dl,
        model_test=FeedForwardModelWithNA_GAN(config, "test", config["tSize_test"], force_var_reuse=True),
        loss_weight_test=None,
        printOnConsole=False, printFreq=1, saveLog=False,
        saveBestFreq=-1, ignoreEpoch=0,
    )
    assert len(sharpe_train) > 0
    assert all(np.isfinite(s) for s in sharpe_train), "NaN/Inf Sharpe in training"
    assert all(np.isfinite(s) for s in sharpe_valid), "NaN/Inf Sharpe in validation"