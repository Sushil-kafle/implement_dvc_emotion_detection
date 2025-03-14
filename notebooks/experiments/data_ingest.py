import kagglehub
from kagglehub import KaggleDatasetAdapter


def get_data(url):
    train_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        url,
        "training.csv",
    )

    test_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        url,
        "test.csv",
    )

    return train_df, test_df
