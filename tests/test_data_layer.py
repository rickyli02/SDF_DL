"""
Tests for DataInRamInputLayer.

These tests are framework-agnostic and should pass against both the TF
and PyTorch implementations without modification.
"""

import numpy as np
import pytest

from src.data.data_layer import DataInRamInputLayer

T, N, F_CHAR, F_MACRO = 12, 30, 46, 178


def test_load_char_only(synthetic_char):
    dl = DataInRamInputLayer(synthetic_char)
    assert dl._return.shape == (T, N)
    assert dl._individualFeature.shape == (T, N, F_CHAR)
    assert dl._mask.shape == (T, N)
    assert dl._mask.dtype == bool


def test_load_with_macro(synthetic_char, synthetic_macro):
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)
    assert dl._macroFeature.shape == (T, F_MACRO)


def test_macro_normalization(synthetic_char, synthetic_macro):
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro, normalizeMacroFeature=True)
    # after z-scoring, each column should have near-zero mean and unit std
    assert np.abs(dl._macroFeature.mean(axis=0)).max() < 1e-5
    np.testing.assert_allclose(dl._macroFeature.std(axis=0), 1.0, atol=1e-5)


def test_macro_normalization_transfer(synthetic_char, synthetic_macro):
    dl_train = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)
    mean, std = dl_train.getMacroFeatureMeanStd()
    dl_valid = DataInRamInputLayer(
        synthetic_char,
        pathMacroFeature=synthetic_macro,
        meanMacroFeature=mean,
        stdMacroFeature=std,
    )
    # applying train stats to identical data should also produce near-zero mean
    assert np.abs(dl_valid._macroFeature.mean(axis=0)).max() < 1e-5


def test_mask_unk_values(synthetic_char):
    dl = DataInRamInputLayer(synthetic_char)
    # all masked-out positions should have return == UNK
    unk_positions = ~dl._mask
    assert (dl._return[unk_positions] == dl._UNK).all()
    # all valid positions should not have return == UNK
    assert (dl._return[dl._mask] != dl._UNK).all()


def test_iterate_one_epoch_shapes(synthetic_char, synthetic_macro):
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)
    batches = list(dl.iterateOneEpoch(subEpoch=False))
    assert len(batches) == 1
    I_macro, I, R, mask = batches[0]
    assert I_macro.shape == (T, F_MACRO)
    assert I.shape == (T, N, F_CHAR)
    assert R.shape == (T, N)
    assert mask.shape == (T, N)


def test_iterate_sub_epoch(synthetic_char, synthetic_macro):
    dl = DataInRamInputLayer(synthetic_char, pathMacroFeature=synthetic_macro)
    SUB = 3
    batches = list(dl.iterateOneEpoch(subEpoch=SUB))
    assert len(batches) == SUB


def test_date_and_variable_accessors(synthetic_char):
    dl = DataInRamInputLayer(synthetic_char)
    assert len(dl.getDateList()) == T
    assert len(dl.getIndividualFeatureList()) == F_CHAR
    # round-trip: index -> var -> index
    for i in range(F_CHAR):
        var = dl.getIndividualFeatureByIdx(i)
        assert dl.getIdxByIndividualFeature(var) == i


def test_date_count_list(synthetic_char):
    dl = DataInRamInputLayer(synthetic_char)
    counts = dl.getDateCountList()
    # each entry is how many non-UNK firms at each date
    assert counts.shape == (N,)
    assert (counts >= 0).all()
    assert (counts <= T).all()