# https://github.com/OpenMined/TenSEAL/issues/307
# https://github.com/OpenMined/TenSEAL/blob/main/tests/python/tenseal/tensors/test_serialization.py
import os
import argparse
import logging
import sys
import glob

import torch
import tenseal as ts
import tenseal.sealapi as sealapi
import base64


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

    parser.add_argument("--input", type=str, required=False)
    parser.add_argument(
        "--extension", type=str, default="bin", help="he file extension"
    )
    parser.add_argument("--output", type=str, required=True)
    # parser.add_argument("--context", type=str, required=True)

    return parser

def initialize_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    context.generate_relin_keys()
    # bits_scale: controls precision of the fractional part
    bits_scale = 40
    # set the scale
    context.global_scale = pow(2, bits_scale)
    return context

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

    if not args.input:
        return 0

    # with open(args.context, "r", encoding="utf-8") as f:
    #     bytes_context = base64.b64decode(f.read())

    # context = ts.context_from(bytes_context)
    context = initialize_context()

    logger.info("Searching for files in {}".format(args.input))
    he_slices = {}
    for f in glob.glob(
        os.path.join(args.input, f"*.{args.extension}"), recursive=False
    ):
        he_slices[os.path.basename(f).rsplit(f".{args.extension}", 1)[0]] = f

    logger.info("Found keys {}".format(list(he_slices.keys())))

    state_dict = {}

    for key in he_slices:
        logger.info("Decrypting layer {}".format(key))
        with open(he_slices[key], "rb") as f:
            bytes_vect = f.read()

        enc_vect = ts.ckks_tensor_from(context, bytes_vect)

        state_dict[key] = torch.Tensor(enc_vect.decrypt().tolist())

    logger.info("Saving model to {}".format(args.output))
    os.makedirs(args.output, exist_ok=True)
    torch.save(state_dict, os.path.join(args.output, "model.pt"))


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
