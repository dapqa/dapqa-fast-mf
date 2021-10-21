import numpy as np
import pytest
from drsu.datasets import as_numpy, MOVIELENS_100K, MOVIELENS_1M, MOVIELENS_10M
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from dfmf.model import SVD, SVDpp
from dfmf.util import sort_by_user

N_ROUNDS = 5


def prepare_random_splits(data, n_splits=1):
    X, y = data[:, 0:2], data[:, 2]

    splits = []
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = sort_by_user(
            *train_test_split(
                X, y,
                test_size=0.2
            )
        )

        splits.append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })

    for split in splits:
        yield split


def do_fit_predict(model_factory, split_generator, benchmark):
    model = model_factory()
    split = next(split_generator)

    model.fit(split['X_train'], split['y_train'])
    y_pred = model.predict(split['X_test'])

    # Saving prediction to benchmark.extra_info to compute errors later (and do not consume extra time at benchmark)
    if 'results' not in benchmark.extra_info:
        benchmark.extra_info['results'] = []

    benchmark.extra_info['results'].append({
        'y_pred': y_pred,
        'y_test': split['y_test']
    })


def calc_prediction_error_stats(benchmark):
    if 'results' not in benchmark.extra_info:
        raise ValueError('Results have not been written to benchmark.extra_info')

    rmse = np.array([
        mean_squared_error(res['y_pred'], res['y_test'], squared=False)
        for res in benchmark.extra_info['results']
    ])

    benchmark.extra_info['rmse_stats'] = {
        'min': np.min(rmse),
        'max': np.max(rmse),
        'mean': np.mean(rmse),
        'stddev': np.std(rmse)
    }
    del benchmark.extra_info['results']


@pytest.mark.benchmark(
    group='svd-ml100k',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svd_ml100k(benchmark, n_factors):
    data = as_numpy(MOVIELENS_100K)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVD(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS), benchmark),
        rounds=N_ROUNDS
    )
    calc_prediction_error_stats(benchmark)


@pytest.mark.benchmark(
    group='svdpp-ml100k',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svdpp_ml100k(benchmark, n_factors):
    data = as_numpy(MOVIELENS_100K)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVDpp(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS), benchmark),
        rounds=N_ROUNDS
    )
    calc_prediction_error_stats(benchmark)


@pytest.mark.benchmark(
    group='svd-ml1m',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svd_ml1m(benchmark, n_factors):
    data = as_numpy(MOVIELENS_1M)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVD(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS), benchmark),
        rounds=N_ROUNDS
    )
    calc_prediction_error_stats(benchmark)


@pytest.mark.benchmark(
    group='svdpp-ml1m',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svdpp_ml1m(benchmark, n_factors):
    data = as_numpy(MOVIELENS_1M)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVDpp(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS), benchmark),
        rounds=N_ROUNDS
    )
    calc_prediction_error_stats(benchmark)


@pytest.mark.benchmark(
    group='svd-ml10m',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svd_ml10m(benchmark, n_factors):
    data = as_numpy(MOVIELENS_10M)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVD(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS), benchmark),
        rounds=N_ROUNDS
    )
    calc_prediction_error_stats(benchmark)


@pytest.mark.benchmark(
    group='svdpp-ml10m',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svdpp_ml10m(benchmark, n_factors):
    data = as_numpy(MOVIELENS_10M)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVDpp(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS), benchmark),
        rounds=N_ROUNDS
    )
    calc_prediction_error_stats(benchmark)
