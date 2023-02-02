"""Script for mock components."""
import argparse
import logging
import sys
import os
import shutil

from azureml.core import Run, Workspace
from azureml.core.keyvault import Keyvault
import splitfolders


def get_kaggle_client(kv: Keyvault):
    """Gets a Kaggle client using the secrets in a key vault to authenticate.

    Args:
        kv (Keyvault): key vault to use for retrieving the Kaggle credentials. The Kaggle user name secret needs to be named 'kaggleusername', while the Kaggle API key secret needs to be named 'kagglekey'.
    """

    os.environ["KAGGLE_USERNAME"] = kv.get_secret("kaggleusername")
    os.environ["KAGGLE_KEY"] = kv.get_secret("kagglekey")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def get_key_vault() -> Keyvault:
    """Retrieves key vault from current run"""
    run: Run = Run.get_context()
    logging.info(f"Got run context: {run}")
    workspace: Workspace = run.experiment.workspace
    return workspace.get_default_keyvault()


def download_unzip_kaggle_dataset(
    kaggle_client, path, dataset_name="paultimothymooney/chest-xray-pneumonia"
):
    """Download a dataset to a specified location and unzip it

    Args:
        kaggle_client (KaggleApi): Instance of KaggleApi to use for retrieving the dataset
        path(str): location where to store downloaded dataset
        dataset_name (str): the name of the dataset to download ('paultimothymooney/chest-xray-pneumonia' by default)
    """
    kaggle_client.dataset_download_files(dataset=dataset_name, path=path, unzip=True)


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    # get the key vault associated to the current workspace
    kv = get_key_vault()

    # authenticate to Kaggle using the secrets in the key vault
    kaggle_client = get_kaggle_client(kv)

    # download and unzip the dataset
    download_unzip_kaggle_dataset(kaggle_client, "./tmp")

    # split it into train, val, and test sets
    splitfolders.ratio(
        input="./tmp/chest_xray/train",
        output="./tmp/chest_xray_tvt/",
        seed=31415,
        ratio=(0.8, 0.1, 0.1),
        group_prefix=None,
        move=True,
    )

    # Create the directories we will populate
    output_path = args.raw_data_folder
    stages = ["train", "val", "test"]
    classes = ["PNEUMONIA", "NORMAL"]
    for stage in stages:
        for class_ in classes:
            os.makedirs(os.path.join(output_path, stage, class_), exist_ok=True)

    # copy extracted jpeg files to output directory
    index = 0
    for root, dirs, files in os.walk("./tmp/chest_xray_tvt"):
        for name in sorted(
            files
        ):  # critical that the list of files is in the same order for each execution, since we partition them into silos based on their position in the list
            if name.endswith(".jpeg"):
                if index % args.silo_count == args.silo_index:
                    shutil.copyfile(
                        os.path.join(root, name),
                        os.path.join(root, name).replace(
                            "./tmp/chest_xray_tvt", output_path
                        ),
                    )
                index += 1


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
        "--raw_data_folder",
        type=str,
        required=True,
        help="Output folder for data",
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
