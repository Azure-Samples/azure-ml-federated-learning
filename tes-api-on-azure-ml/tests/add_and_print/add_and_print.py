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
        "--operand_1",
        required=True,
        type=int,
        help="First operand (an integer).",
    )
    parser.add_argument(
        "--operand_2",
        required=True,
        type=int,
        help="Second operand (an integer).",
    )

    return parser


def main():
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()
    print(f'Operand 1: {args.operand_1}')
    print(f'Operand 2: {args.operand_2}')
    print(
        f'Result: {args.operand_1} + {args.operand_2} = {int(args.operand_1) + int(args.operand_2)}'
    )


if __name__ == "__main__":
    main()
