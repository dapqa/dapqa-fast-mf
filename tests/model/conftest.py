import pytest
from drsu.datasets import as_numpy, MOVIELENS_100K
from sklearn.model_selection import train_test_split

from dfmf.util import sort_by_user


@pytest.fixture(scope='module')
def ml100k_split():
    data = as_numpy(MOVIELENS_100K)
    X, y = data[:, 0:2], data[:, 2]

    X_train, X_test, y_train, y_test = sort_by_user(
        *train_test_split(
            X, y,
            test_size=0.2, random_state=42
        )
    )
    return X_train, X_test, y_train, y_test
