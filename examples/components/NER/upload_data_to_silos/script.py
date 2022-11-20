import argparse
import logging
import sys

from datasets import load_dataset


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
        "--train_data_paths",
        type=str,
        nargs="+",
        required=True,
        help="Output train data paths",
    )
    parser.add_argument(
        "--test_data_paths",
        type=str,
        nargs="+",
        required=True,
        help="Output test data paths",
    )
    return parser


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    df = load_dataset("tner/multinerd", "en", split="test")
    df = df.shuffle(seed=42)

    total_num_of_silos = len(args.train_data_paths)

    for idx in range(total_num_of_silos):

        # partititon dataset
        df_partition = df.shard(num_shards=total_num_of_silos, index=idx)

        # train test split
        df_dict = df_partition.train_test_split(test_size=0.1)

        # save data
        df_dict["train"].save_to_disk(args.train_data_paths[idx])
        df_dict["test"].save_to_disk(args.test_data_paths[idx])


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
