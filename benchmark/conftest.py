import numpy as np
import pytest
from drsu.datasets import download_and_transform_dataset, MOVIELENS_100K, MOVIELENS_1M, MOVIELENS_10M
from drsu.config import DRSUConfiguration

from dfmf.model import SVD, SVDpp


@pytest.fixture(autouse=True, scope='session')
def download_and_transform_data():
    # TODO Add cleanup flag, cleanup will not be performed by default
    # TODO Add option for configuring local_dataset_dir from the run configuration
    DRSUConfiguration.local_dataset_dir = 'data'

    download_and_transform_dataset(MOVIELENS_100K, verbose=False)
    download_and_transform_dataset(MOVIELENS_1M, verbose=False)
    download_and_transform_dataset(MOVIELENS_10M, verbose=False)


@pytest.fixture(autouse=True, scope='session')
def warmup_jit():
    train_data = np.array([
        [1, 1, 5],
        [1, 2, 3],
        [2, 1, 4],
        [2, 3, 5],
        [3, 2, 5],
        [3, 3, 1]
    ])

    test_data = np.array([
        [1, 3, 4],
        [2, 2, 4],
        [3, 1, 4]
    ])

    svd_model = SVD(n_factors=10, n_epochs=1)
    svd_model.fit(train_data[:, :2], train_data[:, 2])
    svd_model.score(test_data[:, :2], test_data[:, 2])

    svdpp_model = SVDpp(n_factors=10, n_epochs=1)
    svdpp_model.fit(train_data[:, :2], train_data[:, 2])
    svdpp_model.score(test_data[:, :2], test_data[:, 2])
