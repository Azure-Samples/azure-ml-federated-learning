"""Script for mock components."""
import argparse
import logging
import sys
import os

import pandas as pd

from zipfile import ZipFile
from azureml.core import Run, Workspace
from azureml.core.keyvault import Keyvault
from azure.ai.ml import MLClient

SPLITS = {
    1: [["Midwest", "Northeast", "South", "West"]],
    2: [["Midwest", "Northeast"], ["South", "West"]],
    3: [["South"], ["Midwest"], ["West", "Northeast"]],
    4: [["South"], ["West"], ["Midwest"], ["Northeast"]],
}


def get_kaggle_client(kv: Keyvault):

    os.environ["KAGGLE_USERNAME"] = kv.get_secret("kaggleusername")
    os.environ["KAGGLE_KEY"] = kv.get_secret("kagglekey")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def get_key_vault() -> Keyvault:
    run: Run = Run.get_context()
    logging.info(f"Got run context: {run}")
    workspace: Workspace = run.experiment.workspace
    return workspace.get_default_keyvault()


def get_ml_client(credential, workspace) -> MLClient:
    subscription_id = workspace.subscription_id
    resource_group = workspace.resource_group
    workspace_name = workspace.name

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    return ml_client


def download_kaggle_dataset(kaggle_client, path):
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
    print(f"Loaded train dataset with {len(df_train)} rows")
    df_test = pd.read_csv("./dataset/extracted/fraudTest.csv", index_col=0)
    print(f"Loaded test dataset with {len(df_train)} rows")
    regions_df = pd.read_csv("./us_regions.csv")
    state_region = {row.StateCode: row.Region for row in regions_df.itertuples()}
    print(f"Loaded state/regions:\n {state_region}")

    df_train.loc[:, "region"] = df_train["state"].map(state_region)
    df_test.loc[:, "region"] = df_test["state"].map(state_region)

    os.makedirs("./dataset/filtered/")
    regions = SPLITS[args.silo_count][args.silo_index]

    print(f"Filtering regions: {regions}")
    train_path = f"{args.raw_train_data}/train.csv"
    test_path = f"{args.raw_test_data}/test.csv"

    df_train_filtered = df_train[df_train["region"].isin(regions)]
    df_test_filtered = df_test[df_test["region"].isin(regions)]
    print(f"Filtered train dataset has {len(df_train_filtered)} rows")
    print(f"Filtered test dataset has {len(df_test_filtered)} rows")

    df_train_filtered.to_csv(train_path, index=False)
    df_test_filtered.to_csv(test_path, index=False)


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
