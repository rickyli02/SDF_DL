"""Tests for model output shapes (PyTorch)."""

import numpy as np
import pytest

from src.data.data_layer import DataInRamInputLayer
from src.model.model_GAN import FeedForwardModelWithNA_GAN

T, N, F_CHAR, F_MACRO = 12, 30, 46, 178


@pytest.fixture(scope="module")
def model_and_data(synthetic_char, synthetic_macro, minimal_config):
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)
    model = FeedForwardModelWithNA_GAN(minimal_config, "test")
    return model, dl


def test_get_weight_shape(model_and_data):
    model, dl = model_and_data
    w = model.getWeightWithData(dl)
    for _, (_, _, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
        expected_len = mask.sum()
    assert w.shape == (expected_len,)
    assert np.isfinite(w).all()


def test_get_sdf_shape(model_and_data):
    model, dl = model_and_data
    sdf = model.getSDF(dl)
    assert sdf.shape == (T, 1)
    assert np.isfinite(sdf).all()


def test_get_sdf_factor_shape(model_and_data):
    model, dl = model_and_data
    factor = model.getSDFFactor(dl)
    assert factor.shape == (T, 1)


def test_normalized_sdf_shape(model_and_data):
    model, dl = model_and_data
    sdf_norm = model.getNormalizedSDF(dl)
    assert sdf_norm.shape == (T, 1)


def test_zero_initial_state_shape(model_and_data, minimal_config):
    model, _ = model_and_data
    state = model.getZeroInitialState()
    hidden = minimal_config["num_units_rnn"][0]
    layers = minimal_config["num_layers_rnn"]
    assert state.shape == (layers, hidden * 2)  # LSTM: h and c concatenated


def test_next_initial_state_shape(model_and_data):
    model, dl = model_and_data
    state0 = model.getZeroInitialState()
    state1 = model.getNextInitialState(dl, state0)
    assert state1.shape == state0.shape
