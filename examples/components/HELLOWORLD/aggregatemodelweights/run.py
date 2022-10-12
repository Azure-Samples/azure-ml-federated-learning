import os
import argparse
import logging
import sys
import glob


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

    parser.add_argument("--input_silo_1", type=str, required=True, help="")
    parser.add_argument("--input_silo_2", type=str, required=False, help="")
    parser.add_argument("--input_silo_3", type=str, required=False, help="")
    parser.add_argument("--aggregated_output", type=str, required=True, help="")
    return parser


def test_input(path):
    file_list = glob.glob(os.path.join(path, "*.*"), recursive=True)
    print(f"Found {len(file_list)} files in {path}")

    print(f"Reading files from {path}")
    for file in file_list:
        print(f" -- Reading {file}")
        with open(file, "r") as f:
            f.read()


def test_output(path):
    print(f"Writing output to {path}/aggregate.txt")
    with open(os.path.join(path, "aggregate.txt"), "w") as f:
        f.write("Hello World!")


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
    test_input(args.input_silo_1)
    if args.input_silo_2:
        test_input(args.input_silo_2)
    if args.input_silo_3:
        test_input(args.input_silo_3)
    test_output(args.aggregated_output)


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
