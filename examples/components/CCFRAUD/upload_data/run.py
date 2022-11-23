"""Script for mock components."""
import argparse
import logging
import sys
import os

import pandas as pd

import azure.ai.ml.identity as identity

from zipfile import ZipFile
from azureml.core import Run, Workspace
from azureml.core.keyvault import Keyvault
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

SPLITS = {
    1: [["Midwest", "Northeast", "South", "West"]],
    2: [["Midwest", "Northeast"], ["South", "West"]],
    3: [["South"], ["Midwest"], ["West", "Northeast"]],
    4: [["South"], ["West"], ["Midwest"], ["Northeast"]],
}


def get_azure_credential() -> AzureMLOnBehalfOfCredential:
    return AzureMLOnBehalfOfCredential()


def get_kaggle_client(kv: Keyvault):

    os.environ["KAGGLE_USERNAME"] = kv.get_secret("kaggleusername")
    os.environ["KAGGLE_KEY"] = kv.get_secret("kagglekey")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def get_workspace() -> Workspace:
    run: Run = Run.get_context()
    logging.info(f"Got run context: {run}")
    workspace: Workspace = run.experiment.workspace
    return workspace


def get_key_vault(workspace) -> Keyvault:
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

    split_count = sum(
        [0 if getattr(args, f"silo_{i}_data", None) is None else 1 for i in range(1, 5)]
    )

    if split_count < 1 or split_count > 4:
        raise Exception("Number of splits/silos must be between 1 and 4 (included)!")

    ws = get_workspace()
    kv = get_key_vault(ws)
    kaggle_client = get_kaggle_client(kv)
    download_kaggle_dataset(kaggle_client, "./dataset")

    with ZipFile("./dataset/fraud-detection.zip", "r") as zObject:
        zObject.extractall("./dataset/extracted")

    df_train = pd.read_csv("./dataset/extracted/fraudTrain.csv")
    print(f"Loaded train dataset with {len(df_train)} rows")
    df_test = pd.read_csv("./dataset/extracted/fraudTrain.csv")
    print(f"Loaded test dataset with {len(df_train)} rows")
    regions_df = pd.read_csv("./us_regions.csv")
    state_region = {row.StateCode: row.Region for row in regions_df.itertuples()}
    print(f"Loaded state/regions:\n {state_region}")

    df_train.loc[:, "region"] = df_train["state"].map(state_region)
    df_test.loc[:, "region"] = df_test["state"].map(state_region)

    os.makedirs("./dataset/filtered/")
    for i, regions in enumerate(SPLITS[split_count]):

        print(f"Filtering regions: {regions}")
        data_path = getattr(args, f"silo_{i + 1}_data", None)
        train_path = f"{data_path}/train.csv"
        test_path = f"{data_path}/test.csv"

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
        "--silo_1_data",
        type=str,
        required=True,
        help="Output folder for silo 1",
    )
    parser.add_argument(
        "--silo_2_data",
        type=str,
        required=False,
        default=None,
        help="Output folder for silo 2",
    )
    parser.add_argument(
        "--silo_3_data",
        type=str,
        required=False,
        default=None,
        help="Output folder for silo 3",
    )
    parser.add_argument(
        "--silo_4_data",
        type=str,
        required=False,
        default=None,
        help="Output folder for silo 4",
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
