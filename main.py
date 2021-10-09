from util.datasets import MOVIELENS_100K, as_numpy, MOVIELENS_1M
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from surprise.prediction_algorithms import SVD
from surprise.dataset import Dataset, Reader

from numba import jit, objmode, types

from numpy.random import default_rng

import numpy as np
import pandas as pd

import time


def predict_naive(X_train, y_train, X_test):
    res = np.full((X_test.shape[0]), np.mean(y_train))
    return res


def predict_surprise(X_train, y_train, X_test):
    train_surprise = Dataset.load_from_df(df=pd.DataFrame(data=np.c_[X_train, y_train.T]),
                                          reader=Reader(rating_scale=(1, 5)))
    train_surprise = train_surprise.build_full_trainset()

    test_surprise = Dataset.load_from_df(df=pd.DataFrame(data=np.c_[X_test, np.zeros(X_test.shape[0]).T]),
                                         reader=Reader(rating_scale=(1, 5)))
    test_surprise = test_surprise.construct_testset(test_surprise.raw_ratings)

    model = SVD()
    model.fit(train_surprise)

    predictions = model.test(test_surprise)
    return np.array([item.est for item in predictions])


def predict_raw_python(X_train, y_train, X_test):
    n_factors = 100
    reg = 0.02
    learning_rate = 0.005
    n_epochs = 20
    rng = np.random.default_rng()

    user_ids_uq = np.unique(X_train[:, 0])
    item_ids_uq = np.unique(X_train[:, 1])

    X_train_widx = np.c_[
        np.searchsorted(user_ids_uq, X_train[:, 0]).T,
        np.searchsorted(item_ids_uq, X_train[:, 1]).T
    ]

    mean_rating = np.mean(y_train)

    user_factors = rng.normal(size=(len(user_ids_uq), n_factors), loc=0, scale=.1)
    item_factors = rng.normal(size=(len(item_ids_uq), n_factors), loc=0, scale=.1)
    user_biases = np.zeros(len(user_ids_uq))
    item_biases = np.zeros(len(item_ids_uq))

    for epoch_number in range(n_epochs):
        # print(f'Epoch #{epoch_number}')
        for i in range(len(X_train_widx)):
            row = X_train_widx[i]
            r_ui = y_train[i]

            u_idx, i_idx = row[0], row[1]
            b_u = user_biases[u_idx]
            b_i = item_biases[i_idx]
            p_u = user_factors[u_idx, :]
            q_i = item_factors[i_idx, :]

            r_ui_pred = mean_rating + b_u + b_i + np.dot(p_u, q_i)

            e_ui = r_ui - r_ui_pred
            user_biases[u_idx] = b_u + learning_rate * (e_ui - reg * b_u)
            item_biases[i_idx] = b_i + learning_rate * (e_ui - reg * b_i)
            user_factors[u_idx] = p_u + learning_rate * (e_ui * q_i - reg * p_u)
            item_factors[i_idx] = q_i + learning_rate * (e_ui * p_u - reg * q_i)

    X_test_widx = np.c_[
        np.searchsorted(user_ids_uq, X_test[:, 0]).T,
        np.searchsorted(item_ids_uq, X_test[:, 1]).T
    ]
    y_pred = np.full((len(X_test_widx)), mean_rating)

    for i in range(len(X_test_widx)):
        row = X_test_widx[i]
        original_row = X_test[i]

        u_idx, i_idx = row[0], row[1]
        if u_idx < 0 or u_idx >= len(user_ids_uq) or user_ids_uq[u_idx] != original_row[0] \
                or i_idx < 0 or i_idx >= len(item_ids_uq) or item_ids_uq[i_idx] != original_row[1]:
            continue

        b_u = user_biases[u_idx]
        b_i = item_biases[i_idx]
        p_u = user_factors[u_idx, :]
        q_i = item_factors[i_idx, :]

        y_pred[i] = mean_rating + b_u + b_i + np.dot(p_u, q_i)

    return y_pred


@jit(types.Array(types.float64, 1, 'C')
         (types.Array(types.int64, 2, 'C'),
          types.Array(types.int64, 2, 'C'),
          types.Array(types.int64, 2, 'C'),
          types.Array(types.int64, 1, 'C'),
          types.Array(types.int64, 1, 'C'),
          types.int64,
          types.int64,
          types.float64,
          types.float64
          ),
     locals={
         'mean_rating': types.float64,
         'user_factors': types.Array(types.float64, 2, 'C'),
         'item_factors': types.Array(types.float64, 2, 'C'),
         'user_biases': types.Array(types.float64, 1, 'C'),
         'item_biases': types.Array(types.float64, 1, 'C'),
         'row': types.Array(types.int64, 1, 'C'),
         'original_row': types.Array(types.int64, 1, 'C'),
         'r_ui': types.int64,
         'u_idx': types.int64,
         'i_idx': types.int64,
         'b_u': types.float64,
         'b_i': types.float64,
         'p_u': types.Array(types.float64, 1, 'C'),
         'q_i': types.Array(types.float64, 1, 'C'),
         'r_ui_pred': types.float64,
         'e_ui': types.float64,
         'y_pred': types.Array(types.float64, 1, 'C'),
     },
     nopython=True, cache=True, fastmath=True)
def predict_numba_inner(train, X_test, X_test_widx, user_ids_uq, item_ids_uq, n_factors, n_epochs,
                        learning_rate, reg):
    with objmode(start_time='f8'):
        start_time = time.perf_counter()

    mean_rating = np.mean(train[:, 2])

    user_factors = np.random.normal(size=(len(user_ids_uq), n_factors), loc=0, scale=.1)
    item_factors = np.random.normal(size=(len(item_ids_uq), n_factors), loc=0, scale=.1)
    user_biases = np.zeros(len(user_ids_uq))
    item_biases = np.zeros(len(item_ids_uq))

    b_u = user_biases[0]
    b_i = item_biases[0]
    p_u = user_factors[0]
    q_i = item_factors[0]
    p_u_buf = np.zeros_like(p_u)
    q_i_buf = np.zeros_like(q_i)
    prev_u_idx = 0
    prev_i_idx = 0
    for epoch_number in range(n_epochs):
        for i in range(len(train)):
            row = train[i]
            u_idx, i_idx, r_ui = row[0], row[1], row[2]

            if prev_u_idx != u_idx:
                user_biases[prev_u_idx] = b_u
                user_factors[prev_u_idx] = p_u
                b_u = user_biases[u_idx]
                p_u = user_factors[u_idx]

            if prev_i_idx != i_idx:
                item_biases[prev_i_idx] = b_i
                item_factors[prev_i_idx] = q_i
                b_i = item_biases[i_idx]
                q_i = item_factors[i_idx]

            r_ui_pred = mean_rating + b_u + b_i + np.dot(p_u, q_i)

            e_ui = r_ui - r_ui_pred
            b_u += learning_rate * (e_ui - reg * b_u)
            b_i += learning_rate * (e_ui - reg * b_i)

            for j in range(n_factors):
                p_u_j = p_u[j]
                q_i_j = q_i[j]
                p_u[j] = p_u_j + learning_rate * (e_ui * q_i_j - reg * p_u_j)
                q_i[j] = q_i_j + learning_rate * (e_ui * p_u_j - reg * q_i_j)

    with objmode():
        print(time.perf_counter() - start_time)

    y_pred = np.full((len(X_test_widx)), mean_rating)

    for i in range(len(X_test_widx)):
        row = X_test_widx[i]
        original_row = X_test[i]

        u_idx, i_idx = row[0], row[1]
        if u_idx < 0 or u_idx >= len(user_ids_uq) or user_ids_uq[u_idx] != original_row[0] \
                or i_idx < 0 or i_idx >= len(item_ids_uq) or item_ids_uq[i_idx] != original_row[1]:
            continue

        b_u = user_biases[u_idx]
        b_i = item_biases[i_idx]
        p_u = user_factors[u_idx, :]
        q_i = item_factors[i_idx, :]

        y_pred[i] = mean_rating + b_u + b_i + np.dot(p_u, q_i)

    return y_pred


def predict_numba(X_train, y_train, X_test):
    n_factors = 100
    reg = 0.02
    learning_rate = 0.005
    n_epochs = 20

    user_ids_uq = np.unique(X_train[:, 0])
    item_ids_uq = np.unique(X_train[:, 1])

    X_train_widx = np.c_[
        np.searchsorted(user_ids_uq, X_train[:, 0]).T,
        np.searchsorted(item_ids_uq, X_train[:, 1]).T
    ]

    X_test_widx = np.c_[
        np.searchsorted(user_ids_uq, X_test[:, 0]).T,
        np.searchsorted(item_ids_uq, X_test[:, 1]).T
    ]

    start_time = time.perf_counter()
    res = predict_numba_inner(
        train=np.c_[X_train_widx, y_train],
        X_test_widx=X_test_widx,
        X_test=X_test,
        user_ids_uq=user_ids_uq,
        item_ids_uq=item_ids_uq,
        n_factors=n_factors,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        reg=reg
    )
    print(time.perf_counter() - start_time)

    return res


if __name__ == '__main__':
    data = as_numpy(MOVIELENS_1M)
    X, y = data[:, 0:2], data[:, 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sort_perm_train = np.argsort(X_train[:, 0])

    X_train = X_train[sort_perm_train, :]
    y_train = y_train[sort_perm_train]

    # start_time = time.perf_counter()
    # y_test_pred_naive = predict_naive(X_train, y_train, X_test)
    # print(f'Surprise: {mean_squared_error(y_test, y_test_pred_naive, squared=False):.4f} RMSE, '
    #       f'{time.perf_counter() - start_time:.6f} seconds')
    #
    # start_time = time.perf_counter()
    # y_test_pred_surprise = predict_surprise(X_train, y_train, X_test)
    # print(f'Surprise: {mean_squared_error(y_test, y_test_pred_surprise, squared=False):.4f} RMSE, '
    #       f'{time.perf_counter() - start_time:.6f} seconds')
    #
    # start_time = time.perf_counter()
    # y_test_pred_raw_python = predict_raw_python(X_train, y_train, X_test)
    # print(f'Raw python: {mean_squared_error(y_test, y_test_pred_raw_python, squared=False):.4f} RMSE, '
    #       f'{time.perf_counter() - start_time:.6f} seconds')

    start_time = time.perf_counter()
    y_test_pred_numba = predict_numba(X_train, y_train, X_test)
    print(f'Numba: {mean_squared_error(y_test, y_test_pred_numba, squared=False):.4f} RMSE, '
          f'{time.perf_counter() - start_time:.6f} seconds')

    # for v, k in predict_numba_inner.inspect_llvm().items():
    #     print(v, k)
