"""
Shared test fixtures.

Creates synthetic .npz files matching the schema expected by DataInRamInputLayer
so tests run without the real (gitignored) datasets.

Schema recap:
  char npz:  data (T, N, F_char+1)  — col 0 is return, cols 1: are features
             date (T,)  variable (F_char+1,)
  macro npz: data (T, F_macro)
             variable (F_macro,)
"""

import json
import numpy as np
import pytest

T, N, F_CHAR, F_MACRO = 12, 30, 46, 178
UNK = -99.99


@pytest.fixture(scope="session")
def synthetic_char(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("char")
    rng = np.random.default_rng(42)

    data = rng.standard_normal((T, N, F_CHAR + 1)).astype("float32")
    # make ~10% returns unknown so mask logic is exercised
    unk_mask = rng.random((T, N)) < 0.1
    data[unk_mask, 0] = UNK

    dates = np.array([f"199{i:02d}01" for i in range(T)])
    variables = np.array(["RET"] + [f"char_{i}" for i in range(F_CHAR)])

    path = tmp / "char.npz"
    np.savez(path, data=data, date=dates, variable=variables)
    return str(path)


@pytest.fixture(scope="session")
def synthetic_macro(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("macro")
    rng = np.random.default_rng(43)

    data = rng.standard_normal((T, F_MACRO)).astype("float32")
    variables = np.array([f"macro_{i}" for i in range(F_MACRO)])

    path = tmp / "macro.npz"
    np.savez(path, data=data, variable=variables)
    return str(path)


@pytest.fixture(scope="session")
def minimal_config():
    return {
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "hidden_dim": [8, 8],
        "hidden_dim_moment": [],
        "num_layers": 2,
        "num_layers_rnn": 1,
        "num_units_rnn": [4],
        "num_layers_rnn_moment": 1,
        "num_units_rnn_moment": [4],
        "num_condition_moment": 4,
        "num_epochs_moment": 2,
        "num_epochs_unc": 2,
        "num_epochs": 2,
        "sub_epoch": 1,
        "use_rnn": True,
        "cell_type_rnn": "lstm",
        "cell_type_rnn_moment": "lstm",
        "num_layers_moment": 0,
        "dropout": 1.0,
        "individual_feature_dim": F_CHAR,
        "macro_feature_dim": F_MACRO,
        "weighted_loss": False,
        "loss_factor": 1.0,
        "macro_idx": None,
        "tSize": T,
        "tSize_valid": T,
        "tSize_test": T,
    }