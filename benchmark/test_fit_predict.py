import pytest
from drsu.datasets import as_numpy, MOVIELENS_100K, MOVIELENS_1M, MOVIELENS_10M
from sklearn.model_selection import train_test_split

from dfmf.model import SVD, SVDpp
from dfmf.util import sort_by_user

N_ROUNDS = 3


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


def do_fit_predict(model_factory, split_generator):
    model = model_factory()
    split = next(split_generator)
    model.fit(split['X_train'], split['y_train'])
    model.predict(split['X_test'])


@pytest.mark.benchmark(
    group='svd-ml100k',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svd_ml100k(benchmark, n_factors):
    data = as_numpy(MOVIELENS_100K)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVD(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS)),
        rounds=N_ROUNDS
    )


@pytest.mark.benchmark(
    group='svdpp-ml100k',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svdpp_ml100k(benchmark, n_factors):
    data = as_numpy(MOVIELENS_100K)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVDpp(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS)),
        rounds=N_ROUNDS
    )


@pytest.mark.benchmark(
    group='svd-ml1m',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svd_ml1m(benchmark, n_factors):
    data = as_numpy(MOVIELENS_1M)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVD(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS)),
        rounds=N_ROUNDS
    )


@pytest.mark.benchmark(
    group='svdpp-ml1m',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svdpp_ml1m(benchmark, n_factors):
    data = as_numpy(MOVIELENS_1M)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVDpp(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS)),
        rounds=N_ROUNDS
    )


@pytest.mark.benchmark(
    group='svd-ml10m',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svd_ml10m(benchmark, n_factors):
    data = as_numpy(MOVIELENS_10M)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVD(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS)),
        rounds=N_ROUNDS
    )


@pytest.mark.benchmark(
    group='svdpp-ml10m',
    disable_gc=True,
)
@pytest.mark.parametrize('n_factors', [10, 100])
def test_svdpp_ml10m(benchmark, n_factors):
    data = as_numpy(MOVIELENS_10M)
    benchmark.pedantic(
        do_fit_predict, args=(lambda: SVDpp(n_factors=n_factors), prepare_random_splits(data, N_ROUNDS)),
        rounds=N_ROUNDS
    )
