import pytest
from drsu.config import DRSUConfiguration
from drsu.datasets import download_and_transform_dataset, MOVIELENS_100K


@pytest.fixture(autouse=True, scope='session')
def download_and_transform_data():
    # TODO Add cleanup flag, cleanup will not be performed by default
    # TODO Add option for configuring local_dataset_dir from the run configuration
    DRSUConfiguration.local_dataset_dir = 'data'

    download_and_transform_dataset(MOVIELENS_100K, verbose=False)
