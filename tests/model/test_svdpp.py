import os

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from dfmf.model import SVDpp


def test_svdpp_fits_and_makes_predictions(ml100k_split):
    X_train, X_test, y_train, y_test = ml100k_split

    model = SVDpp()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == y_test.shape
    assert mean_squared_error(y_test, y_pred, squared=False) < 1


def test_svdpp_properties(ml100k_split):
    X_train, X_test, y_train, y_test = ml100k_split

    model = SVDpp(n_factors=10)
    model.fit(X_train, y_train)
    _ = model.predict(X_test)

    assert model.n_factors == 10
    assert (model.user_ids_uq == np.unique(X_train[:, 0])).all()
    assert (model.item_ids_uq == np.unique(X_train[:, 1])).all()
    assert model.user_factors.shape == (len(model.user_ids_uq), 10)
    assert model.item_factors.shape == (len(model.item_ids_uq), 10)
    assert model.item_imp_factors_sum.shape == (len(model.user_ids_uq), 10)
    assert model.items_of_user_norm.shape == (len(model.user_ids_uq),)
    assert model.user_biases.shape == (len(model.user_ids_uq),)
    assert model.item_biases.shape == (len(model.item_ids_uq),)
    assert model.mean_rating > 0
    
    
def test_svdpp_invalid_fit_input():
    model = SVDpp()

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


def test_svdpp_invalid_predict_input():
    model = SVDpp()
    model.fit(
        np.array([
            [1, 2],
            [2, 3]
        ]),
        np.array([5, 5])
    )

    with pytest.raises(ValueError):
        model.predict(np.array([1, 2]))


def test_sdvpp_jit():
    from dfmf.model._SVDpp import _fit_svdpp, _predict_svdpp
    from numba.core.registry import CPUDispatcher

    assert isinstance(_fit_svdpp, CPUDispatcher)
    assert len(_fit_svdpp.overloads) > 0

    assert isinstance(_predict_svdpp, CPUDispatcher)
    assert len(_predict_svdpp.overloads) > 0


def test_svdpp_save_load(ml100k_split, temp_model_dump_directory):
    X_train, X_test, y_train, y_test = ml100k_split

    model = SVDpp(n_factors=10)
    model.fit(X_train, y_train)

    from joblib import dump, load
    dump_file_name = os.path.join(temp_model_dump_directory, 'test_svdpp_save_load.joblib')
    dump(model, dump_file_name)
    model_from_dump = load(dump_file_name)

    assert isinstance(model_from_dump, SVDpp)
    assert model.n_factors == model_from_dump.n_factors
    assert (model.user_ids_uq == model_from_dump.user_ids_uq).all()
    assert (model.item_ids_uq == model_from_dump.item_ids_uq).all()
    assert (model.user_factors == model_from_dump.user_factors).all()
    assert (model.item_factors == model_from_dump.item_factors).all()
    assert (model.item_imp_factors_sum == model_from_dump.item_imp_factors_sum).all()
    assert (model.items_of_user_norm == model_from_dump.items_of_user_norm).all()
    assert (model.user_biases == model_from_dump.user_biases).all()
    assert (model.item_biases == model_from_dump.item_biases).all()
    assert model.mean_rating == model_from_dump.mean_rating

    y_pred = model.predict(X_test)
    y_pred_by_model_from_dump = model_from_dump.predict(X_test)
    assert (y_pred == y_pred_by_model_from_dump).all()

