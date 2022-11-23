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

    parser.add_argument("--raw_training_data", type=str, required=True, help="")
    parser.add_argument("--raw_testing_data", type=str, required=True, help="")
    parser.add_argument("--train_output", type=str, required=True, help="")
    parser.add_argument("--test_output", type=str, required=True, help="")
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
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
    with open(os.path.join(path, "output.txt"), "w") as f:
        f.write("Hello World!")

def test_local_input_for_external_silos():
    print("This file full path (following symlinks)")
    full_path = os.path.realpath(__file__)
    print(full_path + "\n")

    

    dir_path = "/"

    res=[]
    res_dir = []
    for (dir_path, dir_names, file_names) in os.walk(dir_path):
        res_dir.extend(dir_names)
        res.extend(file_names)
    print("Directories:")
    print(res_dir)
    # print("Files:")
    # print(res)

    with open('/mnt/localdata/data_file.txt') as f:
        lines = f.readlines()
    print("Contents of local input file:")
    print(lines)


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
    test_input(args.raw_training_data)
    test_input(args.raw_testing_data)
    test_output(args.train_output)
    test_output(args.test_output)

    test_local_input_for_external_silos()


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
