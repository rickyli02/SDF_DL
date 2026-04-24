import os
import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.linear_model import LinearRegression


def deco_print(line, end='\n'):
    print('>==================> ' + line, end=end)


def load_dataframe(path, skiprows, nrows):
    df = pd.read_csv(path, skiprows=skiprows, nrows=nrows)
    df.rename(columns={'Unnamed: 0': 'month'}, inplace=True)
    df.set_index('month', inplace=True)
    return df


def sort_by_task_id(folder_list):
    folder_id_list = [(folder, int(folder.split('_')[1])) for folder in folder_list]
    folder_id_list_sorted = sorted(folder_id_list, key=lambda t: t[1])
    return [folder for folder, _ in folder_id_list_sorted]


def Markowitz(r):
    Sigma = r.T.dot(r) / r.shape[0]
    mu = np.mean(r, axis=0)
    w = np.dot(np.linalg.pinv(Sigma), mu)
    return w


def sharpe(r):
    return np.mean(r / r.std())


def load_sorted_results(path, by=None):
    if by is None:
        by = ['sharpe_valid', 'sharpe_test']
    store = pd.HDFStore(path)
    df = store['summary']
    store.close()
    return [df.sort_values(by=[col], ascending=False) for col in by]


def construct_long_short_portfolio(w, R, mask, value=None, low=0.1, high=0.1, normalize=True):
    N_i = np.sum(mask.astype(int), axis=1)
    N_i_cumsum = np.cumsum(N_i)
    w_split = np.split(w, N_i_cumsum)[:-1]
    R_split = np.split(R, N_i_cumsum)[:-1]

    value_weighted = False
    if value is not None:
        value_weighted = True
        value_split = np.split(value, N_i_cumsum)[:-1]

    portfolio_returns = []
    for j in range(mask.shape[0]):
        R_j = R_split[j]
        w_j = w_split[j]
        if value_weighted:
            value_j = value_split[j]
            R_w_j = [(R_j[k], w_j[k], value_j[k]) for k in range(N_i[j])]
        else:
            R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])]
        R_w_j_sorted = sorted(R_w_j, key=lambda t: t[1])
        n_low = int(low * N_i[j])
        n_high = int(high * N_i[j])

        if n_high == 0:
            portfolio_return_high = 0.0
        else:
            portfolio_return_high = 0.0
            value_sum_high = 0.0
            for k in range(n_high):
                portfolio_return_high += R_w_j_sorted[-k - 1][0] * R_w_j_sorted[-k - 1][2]
                value_sum_high += R_w_j_sorted[-k - 1][2]
            if normalize:
                portfolio_return_high /= value_sum_high

        if n_low == 0:
            portfolio_return_low = 0.0
        else:
            portfolio_return_low = 0.0
            value_sum_low = 0.0
            for k in range(n_low):
                portfolio_return_low += R_w_j_sorted[k][0] * R_w_j_sorted[k][2]
                value_sum_low += R_w_j_sorted[k][2]
            if normalize:
                portfolio_return_low /= value_sum_low

        portfolio_returns.append(portfolio_return_high - portfolio_return_low)
    return np.array(portfolio_returns)