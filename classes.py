import time

import numpy as np
from drsu.datasets import MOVIELENS_100K, as_numpy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from dfmf.model import SVDpp

if __name__ == '__main__':
    data = as_numpy(MOVIELENS_100K)
    X, y = data[:, 0:2], data[:, 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sort_perm_train = np.argsort(X_train[:, 0])
    sort_perm_test = np.argsort(X_test[:, 0])

    X_train = X_train[sort_perm_train, :]
    y_train = y_train[sort_perm_train]

    X_test = X_test[sort_perm_test, :]
    y_test = y_test[sort_perm_test]

    # ---
    start_time = time.perf_counter()

    model = SVDpp()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter()

    y_test_pred = model.predict(X_test)
    predict_time = time.perf_counter()

    print(f'SVD++: {mean_squared_error(y_test, y_test_pred, squared=False):.4f} RMSE,\n'
          f'fit in {fit_time - start_time:.6f} s,\n'
          f'predict in {predict_time - fit_time:.6f} s,\n'
          f'total {predict_time - start_time:.6f} s,\n')
