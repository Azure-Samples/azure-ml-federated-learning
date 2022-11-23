"""Script for mock components."""
import argparse
import logging
import sys
import os

import pandas as pd

import azure.ai.ml.identity as identity

from zipfile import ZipFile
from azureml.core import Run, Workspace
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
    client_id = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID')
    credential = ManagedIdentityCredential(client_id=client_id)
    return credential #AzureMLOnBehalfOfCredential()

def get_kaggle_client(credential):
    secret_client = SecretClient(vault_url="https://kv-aml-fldemodm0.vault.azure.net/", credential=credential)
    
    print(secret_client.get_secret("kaggleusername").value)
    print(secret_client.get_secret("kagglekey").value)

    os.environ['KAGGLE_USERNAME'] = secret_client.get_secret("kaggleusername").value
    os.environ['KAGGLE_KEY'] = secret_client.get_secret("kagglekey").value

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def get_ml_client(credential) -> MLClient:
    run: Run = Run.get_context()
    logging.info(f"Got run context: {run}")
    workspace: Workspace = run.experiment.workspace

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

    if args.split_count < 1 or args.split_count > 4:
        raise Exception("Number of splits must be between 1 and 4 (included)!")

    credential = get_azure_credential()
    kaggle_client = get_kaggle_client(credential)
    download_kaggle_dataset(kaggle_client, "./dataset")

    with ZipFile("./dataset/fraud-detection.zip", "r") as zObject:
        zObject.extractall('./dataset/extracted')

    df_train = pd.read_csv("./dataset/extracted/fraudTrain.csv")
    df_test = pd.read_csv("./dataset/extracted/fraudTrain.csv")
    regions_df = pd.read_csv("./us_regions.csv")
    state_region = {row.StateCode: row.Region for row in regions_df.itertuples()}

    df_train.loc[:, "region"] = df_train["state"].map(state_region)
    df_test.loc[:, "region"] = df_test["state"].map(state_region)

    ml_client = get_ml_client(credential)
    os.makedirs("./dataset/filtered/")
    for i, regions in enumerate(SPLITS[args.split_count]):
        train_name = f"cc_fraud_train_split_{i}"
        test_name = f"cc_fraud_test_split_{i}"
        train_path = f"./dataset/filtered/{train_name}.csv"
        test_path = f"./dataset/filtered/{test_name}.csv"

        df_train_filtered = df_train[df_train["state"].isin(regions)]
        df_test_filtered = df_test[df_test["state"].isin(regions)]

        df_train_filtered.to_csv(train_path, index=False)
        df_test_filtered.to_csv(test_path, index=False)

        ds_name = args.target_ds.split(",")[i]
        datastore = ml_client.datastores.get(ds_name)
        # target_path = f"azureml://datastores/{ds_name}/paths/cc_fraud/{train_name}"
        train_dataset = Data(
            path=train_path,
            type=AssetTypes.URI_FILE,
            name=train_name,
            datastore=datastore
        )
        ml_client.data.create_or_update(train_dataset)
        test_dataset = Data(
            path=test_path,
            type=AssetTypes.URI_FILE,
            name=test_name,
            datastore=datastore
        )
        ml_client.data.create_or_update(test_dataset)



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

    parser.add_argument("--split_count", default=3, type=int, required=False, help="Batch Size")
    parser.add_argument("--target_ds", type=str, required=True, help="Comma separated names of storage accounts to use for datasets")
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
