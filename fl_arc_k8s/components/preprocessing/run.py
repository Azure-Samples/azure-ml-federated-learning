"""run.py for mock components"""
import argparse


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse

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
        "--raw_data",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--train_output",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--test_output",
        type=str,
        required=True,
        help=""
    )
    return parser


def run(args):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
    """
    with(open(os.path.join(args.train_output, "output.txt"), "w")) as out_file:
        out_file.write("fake training output")

    with(open(os.path.join(args.test_output, "output.txt"), "w")) as out_file:
        out_file.write("fake testing output")


def main(cli_args=None):
    """ Component main function.
    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    print(f"Running script with arguments: {args}")
    run(args)


if __name__ == "__main__":
    main()
