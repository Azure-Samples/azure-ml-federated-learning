"""Script for mock components."""
import argparse
import logging
import sys
import os

import numpy as np
import pandas as pd

from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from zipfile import ZipFile
from azureml.core import Run, Workspace
from azureml.core.keyvault import Keyvault

CATEGORICAL_NOMINAL_PROPS = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
CATEGORICAL_ORDINAL_PROPS = ["month", "day_of_week"]
ORDINAL_CATEGORIES = {
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
}
NUMERICAL_PROPS = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
ENCODERS = {}
SCALERS = {}

SPLITS = {
    2: [["age", "job", "marital", "education", "default", "housing", "loan"], ["contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]],
    3: [["age", "job", "marital", "education", "default", "housing", "loan"], ["contact", "month", "day_of_week", "duration"], ["campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]],
    4: [["age", "job", "marital", "education", "default", "housing", "loan"], ["contact", "month", "day_of_week", "duration"], ["campaign", "pdays", "previous", "poutcome"], ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]],
}


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

    for column in df.columns:
        if column in CATEGORICAL_NOMINAL_PROPS and column not in ENCODERS:
            print(f"Creating encoder for column: {column}")
            # Simply set all zeros if the category is unseen
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(df[column].values.reshape(-1, 1))
            ENCODERS[column] = encoder

        if column in CATEGORICAL_NOMINAL_PROPS and column not in ENCODERS:
            print(f"Creating encoder for column: {column}")
            # Simply set all zeros if the category is unseen
            encoder = FeatureHasher(input_type="string", n_features=4)
            ENCODERS[column] = encoder

        if column in CATEGORICAL_ORDINAL_PROPS and column not in ENCODERS:
            print(f"Creating encoder for column: {column}")
            # Simply set all zeros if the category is unseen
            encoder = OrdinalEncoder(categories=[ORDINAL_CATEGORIES[column]])
            encoder.fit(df[column].values.reshape(-1, 1))
            ENCODERS[column] = encoder

        if column in NUMERICAL_PROPS and column not in SCALERS:
            print(f"Creating scaler for column: {column}")
            scaler = StandardScaler()
            scaler.fit(df[column].values.reshape(-1, 1))
            SCALERS[column] = scaler


def split_load(df, silo_count):
    loads = {col: 1 for col in df.columns}
    for category in ENCODERS:
        if category in loads:
            if hasattr(ENCODERS[category], "get_feature_names"):
                loads[category] = len(ENCODERS[category].get_feature_names())
            elif hasattr(ENCODERS[category], "categories"):
                loads[category] = len(ENCODERS[category].categories[0])
            else:
                loads[category] = ENCODERS[category].n_features

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

    for column in ENCODERS:
        if column not in df.columns:
            continue

        encoder = ENCODERS.get(column)
        print(f"Transforming column: {column}")
        encoded_data = encoder.transform(
            df[column].astype(str).values.reshape(-1, 1)
        )
        df.drop(column, axis=1, inplace=True)

        if type(encoder) == OrdinalEncoder:
            df[column] = encoded_data[:, 0]
        elif type(encoder) == OneHotEncoder:
            encoded_data = encoded_data.toarray()
            for i, name in enumerate(encoder.get_feature_names()):
                df[column + "_" + "_".join(name.split("_")[1:])] = encoded_data[:, i]
        else:
            encoded_data = encoded_data.toarray()
            for i in range(encoder.n_features):
                df[f"{column}_{i}"] = encoded_data[:, i]

    for column in SCALERS:
        if column not in df.columns:
            continue

        scaler = SCALERS.get(column)
        df[column] = scaler.transform(df[column].values.reshape(-1, 1))
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
    kaggle_client.dataset_download_files("volodymyrgavrysh/bank-marketing-campaigns-dataset", path=path)


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

    with ZipFile("./dataset/bank-marketing-campaigns-dataset.zip", "r") as zObject:
        zObject.extractall("./dataset/extracted")

    df = pd.read_csv("./dataset/extracted/bank-additional-full.csv", sep=";")
    logger.info(f"{df.head()}")
    train_size = int(len(df) * 0.8)
    df_train = df[:train_size]
    df_test = df[train_size:]

    print(f"Loaded train dataset with {len(df_train)} rows")
    print(f"Loaded test dataset with {len(df_test)} rows")

    os.makedirs(args.raw_train_data, exist_ok=True)
    os.makedirs(args.raw_test_data, exist_ok=True)

    train_path = f"{args.raw_train_data}/data.csv"
    test_path = f"{args.raw_test_data}/data.csv"

    label_train = df_train["y"]
    label_test = df_test["y"]

    df_train.drop("y", axis=1, inplace=True)
    df_test.drop("y", axis=1, inplace=True)

    # # Create categorical encoder before any further preprocessing/reduction
    fit_encoders(df_train)
    column_splits = SPLITS[int(args.silo_count)] # split_load(df_train, int(args.silo_count))

    drop_columns_subset = set(df_train.columns).difference(
        column_splits[args.silo_index]
    )
    df_train.drop(drop_columns_subset, 1, inplace=True)
    df_test.drop(drop_columns_subset, 1, inplace=True)

    preprocess_data(df_train)
    preprocess_data(df_test)

    if args.silo_index == 0:
        df_train["label"] = [1 if label == "yes" else 0 for label in label_train]
        df_test["label"] = [1 if label == "yes" else 0 for label in label_test]

        subscribe_weight = (
            df_train["label"].value_counts()[0] / df_train["label"].value_counts()[1]
        )
        logger.debug(f"Subscribe weight: {subscribe_weight}")
        np.savetxt(args.raw_train_data + "/subscribe_weight.txt", np.array([subscribe_weight]))

    logger.info(f"Train data head columns: {df_train.columns}")
    logger.info("Train data head:\n")
    logger.info(df_train.head())

    logger.info(f"Test data head columns: {df_test.columns}")
    logger.info("Test data head:\n")
    logger.info(df_test.head())

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
