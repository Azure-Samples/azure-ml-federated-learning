"""Script for mock components."""
import argparse
import logging
import sys

import pandas as pd

DATASET_URL = {
    "train": "https://azureopendatastorage.blob.core.windows.net/mnist/processed/train.csv",
    "test": "https://azureopendatastorage.blob.core.windows.net/mnist/processed/t10k.csv"
}

def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    if args.silo_count != 2 or (args.silo_index != 0 and args.silo_index != 1):
        raise Exception("Number of splits/silos must be 2 and silo index must be either 0 or 1!")

    for x in ["train", "test"]:
        attrname = f"raw_{x}_data"
        store_path = f"{getattr(args, attrname)}/{x}.csv"
        df = pd.read_csv(DATASET_URL["train"], index_col=0).reset_index()

        if args.silo_index == 0:
            df[["label"]].to_csv(store_path)
        else:
            df = df.loc[:, df.columns != "label"]
            df.to_csv(store_path)


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
