"""Script for mock components."""
import argparse
import logging
import sys
import os

from datasets import load_dataset


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    df = load_dataset("tner/multinerd", "en", split="test")
    df = df.shuffle(seed=42)

    # partititon dataset
    df_partition = df.shard(num_shards=args.silo_count, index=args.silo_index)

    # train test split
    df_dict = df_partition.train_test_split(test_size=0.1)

    # save data
    df_dict["train"].save_to_disk(args.raw_train_data)
    df_dict["test"].save_to_disk(args.raw_test_data)


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
