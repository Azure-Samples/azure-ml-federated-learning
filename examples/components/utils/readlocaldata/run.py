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

    parser.add_argument("--local_data_path_input", type=str, required=True, help="")
    parser.add_argument(
        "--preprocessed_local_data_output", type=str, required=True, help=""
    )
    return parser


def run(args):
    """Component run function. This will read the local data, preproces them, and write the preprocessed data to the output."""
    # Read the contents of the local data file
    with open(os.path.join(args.local_data_path_input, "data_file.txt")) as in_f:
        lines = in_f.readlines()
    print("Contents of local input file:")
    print(lines)  # Be careful here, you don't want to print sensitive user data!

    # "preprocess" the data (simple conversion to lower case)
    preprocessed_lines = [line.lower() for line in lines]
    print("Preprocessed data:")
    print(
        preprocessed_lines
    )  # Be careful here, you don't want to print sensitive user data!

    # write the preprocessed data to the output
    with open(
        os.path.join(args.preprocessed_local_data_output, "data_file.txt")
    ) as out_f:
        out_f.writelines(preprocessed_lines)


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
