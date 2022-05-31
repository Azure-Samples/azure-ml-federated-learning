import os
from argparse import ArgumentParser


def get_arg_parser(parser=None) -> ArgumentParser:
    """Parse the command line arguments for merge using argparse

    Args:
        parser (ArgumentParser): an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    if parser is None:
        parser = ArgumentParser(description=__doc__)

    # Input parameters
    parser.add_argument(
        "--argument_1",
        required=True,
        type=str,
        help="Example argument that we'll just print out.",
    )
    parser.add_argument(
        "--input_data_1",
        required=True,
        type=str,
        help="First input dataset.",
    )
    parser.add_argument(
        "--input_data_2",
        required=True,
        type=str,
        help="Second input dataset.",
    )

    return parser


def main():
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()
    print(f"Argument 1: {args.argument_1}")
    print(f"First input dataset has {len(os.listdir(args.input_data_1))} file(s).")
    print(f"Second input dataset has {len(os.listdir(args.input_data_2))} file(s).")


if __name__ == "__main__":
    main()
