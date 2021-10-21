import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from dfmf.model import SVD


def test_svd_fits_and_makes_predictions(ml100k_split):
    X_train, X_test, y_train, y_test = ml100k_split

    model = SVD()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == y_test.shape
    assert mean_squared_error(y_test, y_pred, squared=False) < 1


def test_svd_properties(ml100k_split):
    X_train, X_test, y_train, y_test = ml100k_split

    model = SVD(n_factors=10)
    model.fit(X_train, y_train)
    _ = model.predict(X_test)

    assert model.n_factors == 10
    assert (model.user_ids_uq == np.unique(X_train[:, 0])).all()
    assert (model.item_ids_uq == np.unique(X_train[:, 1])).all()
    assert model.user_factors.shape == (len(model.user_ids_uq), 10)
    assert model.item_factors.shape == (len(model.item_ids_uq), 10)
    assert model.user_biases.shape == (len(model.user_ids_uq),)
    assert model.item_biases.shape == (len(model.item_ids_uq),)
    assert model.mean_rating > 0


def test_svd_invalid_fit_input():
    model = SVD()

    with pytest.raises(ValueError):
        model.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        model.fit(
            np.array([
                [1, 2],
                [2, 3]
            ]),
            np.array([
                [1, 2],
                [2, 3]
            ])
        )

    with pytest.raises(ValueError):
        model.fit(
            np.array([
                [1, 2],
                [2, 3]
            ]),
            np.array([1, 2, 3])
        )


def test_svd_invalid_predict_input():
    model = SVD()
    model.fit(
        np.array([
            [1, 2],
            [2, 3]
        ]),
        np.array([5, 5])
    )

    with pytest.raises(ValueError):
        model.predict(np.array([1, 2]))


def test_svd_jit():
    from dfmf.model._SVD import _fit_svd, _predict_svd
    from numba.core.registry import CPUDispatcher

    assert isinstance(_fit_svd, CPUDispatcher)
    assert len(_fit_svd.overloads) > 0

    assert isinstance(_predict_svd, CPUDispatcher)
    assert len(_predict_svd.overloads) > 0


