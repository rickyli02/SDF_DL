"""Tests that a single training run completes without NaN losses (PyTorch)."""

import numpy as np
import pytest

from src.data.data_layer import DataInRamInputLayer
from src.model.model_GAN import FeedForwardModelWithNA_GAN


def test_one_training_run_no_nan(synthetic_char, synthetic_macro, minimal_config, tmp_path):
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)
    model = FeedForwardModelWithNA_GAN(minimal_config, "train")

    result = model.train_model(
        dl, dl,
        logdir=str(tmp_path / "ckpt"),
        loss_weight=None, loss_weight_valid=None,
        dl_test=dl, loss_weight_test=None,
        printOnConsole=False,
        printFreq=9999,
        saveLog=False,
        saveBestFreq=-1,
        ignoreEpoch=0,
    )
    sharpe_train, sharpe_valid, _ = result
    assert len(sharpe_train) > 0
    assert all(np.isfinite(s) for s in sharpe_train), f"NaN in train Sharpe: {sharpe_train}"
    assert all(np.isfinite(s) for s in sharpe_valid), f"NaN in valid Sharpe: {sharpe_valid}"


def test_checkpoint_save_and_load(synthetic_char, synthetic_macro, minimal_config, tmp_path):
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)
    model = FeedForwardModelWithNA_GAN(minimal_config, "train")

    model.train_model(
        dl, dl,
        logdir=str(tmp_path / "ckpt"),
        printOnConsole=False, printFreq=9999,
        saveLog=False, saveBestFreq=-1, ignoreEpoch=0,
    )

    # Load from the sharpe subdir and check inference still works
    logdir_sharpe = str(tmp_path / "ckpt" / "sharpe")
    model2 = FeedForwardModelWithNA_GAN(minimal_config, "test")
    model2.load(logdir_sharpe)
    sdf = model2.getSDF(dl)
    assert sdf.shape[0] > 0
    assert np.isfinite(sdf).all()
