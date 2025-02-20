import pandas as pd
import os
import dvc.api
import kagglehub
from kagglehub import KaggleDatasetAdapter

from dotenv import load_dotenv

load_dotenv()


params = dvc.api.params_show().get("data_ingestion", {})


def fetch_data(url: str) -> tuple:
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

    return (train_df, test_df)


def save_data(
    data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> None:
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)


url = params["url"]


def main():
    train_df, test_df = fetch_data(url)
    data_path = os.path.join("data", "raw")
    save_data(data_path, train_df, test_df)


if __name__ == "__main__":
    main()
