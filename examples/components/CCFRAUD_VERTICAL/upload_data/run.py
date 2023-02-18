"""Script for mock components."""
import argparse
import logging
import sys
import os

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from zipfile import ZipFile
from azureml.core import Run, Workspace
from azureml.core.keyvault import Keyvault

CATEGORICAL_PROPS = ["category", "region", "gender", "state"]
ENCODERS = {}


def get_kaggle_client(kv: Keyvault):
    """Gets the Kaggle client

    Args:
        kv (Keyvault): keyvault to use for retrieving Kaggle credentials
    """

    os.environ["KAGGLE_USERNAME"] = kv.get_secret("kaggleusername")
    os.environ["KAGGLE_KEY"] = kv.get_secret("kagglekey")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def fit_encoders(df):
    """Creates one-hot encodings for categorical data

    Args:
        df (pd.DataFrame): Pandas dataframe to use to provide us with all unique value for each categorical column
    """

    global ENCODERS

    for column in CATEGORICAL_PROPS:
        if column not in df.columns:
            continue

        if column not in ENCODERS:
            print(f"Creating encoder for column: {column}")
            # Simply set all zeros if the category is unseen
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(df[column].values.reshape(-1, 1))
            ENCODERS[column] = encoder


def filter_useful_columns(df: pd.DataFrame):
    useful_props = [
        "amt",
        "age",
        "merch_lat",
        "merch_long",
        "category",
        "region",
        "gender",
        "state",
        "lat",
        "long",
        "city_pop",
        "trans_date_trans_time",
        "is_fraud",
    ]

    # Filter only useful columns
    df.drop(set(df.columns).difference(useful_props), 1, inplace=True)
    return df


def split_load(df, silo_count):
    loads = {col: 1 for col in df.columns}
    for category in CATEGORICAL_PROPS:
        if category in loads:
            loads[category] = len(ENCODERS.get(category).get_feature_names())

    target_load = sum(loads.values()) / silo_count
    sorted_loads = sorted(loads.items(), key=lambda x: x[1], reverse=True)

    print("Sorted loads:", sorted_loads)
    print("Silo count:", silo_count)

    current_load = 0
    split, splits = [], []
    for column, load in sorted_loads:
        split.append(column)
        current_load += load

        if (current_load >= target_load and len(splits) < silo_count - 1) or len(
            sorted_loads
        ) - sorted_loads.index((column, load)) - 1 <= silo_count - len(splits) - 1:
            splits.append(split)
            split = []
            current_load = 0

    if len(split) > 0:
        splits.append(split)

    print("Splits:", splits)
    return splits


def preprocess_data(df: pd.DataFrame):
    """Filter dataframe to include only useful features and apply categorical one hot encoders

    Args:
        df (pd.DataFrame): Pandas dataframe to apply transforms to
    """

    for column in CATEGORICAL_PROPS:
        if column not in df.columns:
            continue

        encoder = ENCODERS.get(column)
        encoded_data = encoder.transform(df[column].values.reshape(-1, 1)).toarray()
        df.drop(column, axis=1, inplace=True)
        for i, name in enumerate(encoder.get_feature_names()):
            df[column + "_" + "_".join(name.split("_")[1:])] = encoded_data[:, i]
    return df


def get_key_vault() -> Keyvault:
    """Retreives keyvault from current run"""
    run: Run = Run.get_context()
    logging.info(f"Got run context: {run}")
    workspace: Workspace = run.experiment.workspace
    return workspace.get_default_keyvault()


def download_kaggle_dataset(kaggle_client, path):
    """Downloads datasets to specified location

    Args:
        kaggle_client (KaggleApi): Instance of KaggleApi to use for retrieving the dataset
        path(str): location where to store downloaded dataset
    """
    kaggle_client.dataset_download_files("kartik2112/fraud-detection", path=path)


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    if args.silo_count < 1 or args.silo_count > 4:
        raise Exception("Number of splits/silos must be between 1 and 4 (included)!")

    kv = get_key_vault()
    kaggle_client = get_kaggle_client(kv)
    download_kaggle_dataset(kaggle_client, "./dataset")

    with ZipFile("./dataset/fraud-detection.zip", "r") as zObject:
        zObject.extractall("./dataset/extracted")

    df_train = pd.read_csv("./dataset/extracted/fraudTrain.csv", index_col=0)
    df_train = df_train.sort_values(by="trans_date_trans_time")
    df_train.reset_index(inplace=True)
    print(f"Loaded train dataset with {len(df_train)} rows")
    df_test = pd.read_csv("./dataset/extracted/fraudTest.csv", index_col=0)
    df_test = df_test.sort_values(by="trans_date_trans_time")
    df_test.reset_index(inplace=True)
    print(f"Loaded test dataset with {len(df_test)} rows")

    if not os.path.exists(args.raw_train_data):
        os.makedirs(args.raw_train_data, exist_ok=True)
    if not os.path.exists(args.raw_test_data):
        os.makedirs(args.raw_test_data, exist_ok=True)

    train_path = f"{args.raw_train_data}/train.csv"
    test_path = f"{args.raw_test_data}/test.csv"

    if args.silo_index == 0:
        df_train = df_train[["is_fraud"]]
        df_test = df_test[["is_fraud"]]
    else:
        df_train.drop("is_fraud", axis=1, inplace=True)
        df_test.drop("is_fraud", axis=1, inplace=True)

        df_train.loc[:, "age"] = (
            pd.Timestamp.now() - pd.to_datetime(df_train["dob"])
        ) // pd.Timedelta("1y")
        df_test.loc[:, "age"] = (
            pd.Timestamp.now() - pd.to_datetime(df_test["dob"])
        ) // pd.Timedelta("1y")
        filter_useful_columns(df_train)
        filter_useful_columns(df_test)

        regions_df = pd.read_csv("./us_regions.csv")
        state_region = {row.StateCode: row.Region for row in regions_df.itertuples()}
        print(f"Loaded state/regions:\n {state_region}")

        df_train.loc[:, "region"] = df_train["state"].map(state_region)
        df_test.loc[:, "region"] = df_test["state"].map(state_region)

        print(
            f"Train dataset has {len(df_train)} rows and {len(df_train.columns)} columns: {list(df_train.columns)}"
        )
        print(
            f"Test dataset has {len(df_test)} rows and {len(df_test.columns)} columns: {list(df_test.columns)}"
        )

        # Create categorical encoder before any further preprocessing/reduction
        fit_encoders(df_train)
        column_splits = split_load(df_train, int(args.silo_count) - 1)

        drop_columns_subset = set(df_train.columns).difference(
            column_splits[args.silo_index - 1]
        )

        df_train.drop(drop_columns_subset, 1, inplace=True)
        df_test.drop(drop_columns_subset, 1, inplace=True)

    preprocess_data(df_train)
    preprocess_data(df_test)

    print(
        f"Filtered train dataset has {len(df_train)} rows and {len(df_train.columns)} columns: {list(df_train.columns)}"
    )
    print(
        f"Filtered test dataset has {len(df_test)} rows and {len(df_test.columns)} columns: {list(df_test.columns)}"
    )

    df_train.to_csv(train_path)
    df_test.to_csv(test_path)


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse.

    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser):
        an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the component
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--silo_count",
        type=int,
        required=True,
        help="Number of silos",
    )
    parser.add_argument(
        "--silo_index",
        type=int,
        required=True,
        help="Index of the current silo",
    )
    parser.add_argument(
        "--raw_train_data",
        type=str,
        required=True,
        help="Output folder for train data",
    )
    parser.add_argument(
        "--raw_test_data",
        type=str,
        required=True,
        help="Output folder for test data",
    )
    return parser


def main(cli_args=None):
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """

    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")
    run(args)


if __name__ == "__main__":
    # Set logging to sys.out
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    main()
