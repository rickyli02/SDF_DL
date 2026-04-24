import numpy as np
import torch.nn as nn


def build_rnn(cell_type, input_size, num_units, num_layers=1, dropout=0.0):
    """Construct a ready-to-use RNN with the correct input_size."""
    hidden_size = num_units[0]
    inter_dropout = dropout if num_layers > 1 else 0.0
    cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}.get(cell_type)
    if cls is None:
        raise ValueError(f"Cell type '{cell_type}' not supported")
    return cls(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=inter_dropout,
        batch_first=True,
    )


def getFactor(beta, dl, normalized=False, norm='l2'):
    for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
        R_reshape = R[mask]
        splits = np.sum(mask, axis=1).cumsum()[:-1]
        beta_list = np.split(beta, splits)
        R_list = np.split(R_reshape, splits)
        F_list = []
        for R_i, beta_i in zip(R_list, beta_list):
            if normalized:
                if norm == 'l1':
                    F_list.append(R_i.dot(beta_i) / np.absolute(beta_i).sum())
                else:
                    F_list.append(R_i.dot(beta_i) / np.sqrt(beta_i.dot(beta_i)))
            else:
                F_list.append(R_i.dot(beta_i) / beta_i.dot(beta_i))
        return np.array(F_list)


def decomposeReturn(w, dl):
    for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
        R_reshape = R[mask]
        splits = np.sum(mask, axis=1).cumsum()[:-1]
        w_list = np.split(w, splits)
        R_list = np.split(R_reshape, splits)
        R_hat_list = []
        residual_list = []
        for R_i, w_i in zip(R_list, w_list):
            R_hat_i = w_i.dot(R_i) / w_i.dot(w_i) * w_i
            residual_i = R_i - R_hat_i
            R_hat_list.append(R_hat_i)
            residual_list.append(residual_i)
        R_hat = np.zeros_like(mask, dtype=float)
        residual = np.zeros_like(mask, dtype=float)
        R_hat[mask] = np.concatenate(R_hat_list)
        residual[mask] = np.concatenate(residual_list)
    return R_hat, residual, mask, R


def calculateStatistics(w, dl):
    R_hat, residual, mask, R = decomposeReturn(w, dl)
    T_i = np.sum(mask, axis=0)
    N_t = np.sum(mask, axis=1)
    stat1 = 1 - np.mean(np.square(residual).sum(axis=1) / N_t) / np.mean(np.square(R * mask).sum(axis=1) / N_t)
    stat2 = 1 - np.mean(np.square(residual.sum(axis=0) / T_i)) / np.mean(np.square((R * mask).sum(axis=0) / T_i))
    stat3 = 1 - np.mean(np.square(residual.sum(axis=0) / T_i) * T_i) / np.mean(np.square((R * mask).sum(axis=0) / T_i) * T_i)
    return stat1, stat2, stat3