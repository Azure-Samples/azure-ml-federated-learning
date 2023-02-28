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

    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        nargs="+",
        help="list of paths or directories to search for he vector files",
    )
    # parser.add_argument(
    #     "--extension", type=str, default="bin", help="he file extension"
    # )
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

def run(checkpoint_folders, output_folder):
    logger = logging.getLogger(__name__)
    context = initialize_context()
    hf_files_paths = []
    for model_path in checkpoint_folders:
        logger.info("Searching for files {}".format(model_path))
        for f in glob.glob(os.path.join(model_path, "**"), recursive=True):
            if f.endswith("bin"):
                hf_files_paths.append(f)

    logger.info("Found {} he files".format(len(hf_files_paths)))

    model_slices = {}

    for file_path in hf_files_paths:
        slice_name = os.path.basename(file_path)
        if slice_name not in model_slices:
            model_slices[slice_name] = []
        model_slices[slice_name].append(file_path)

    logger.info(
        "Found {} slices: {}".format(len(model_slices), list(model_slices.keys()))
    )

    for model_slice in model_slices.keys():
        agg_slice = None

        for i in range(len(model_slices[model_slice])):
            logger.info(
                "Aggregating slice {} {}/{}".format(
                    model_slice, i, len(model_slices[model_slice])
                )
            )
            with open(model_slices[model_slice][i], "rb") as f:
                bytes_vect = f.read()

            enc_vect = ts.ckks_tensor_from(context, bytes_vect)
            # enc_vect = bytes_vect

            if agg_slice is None:
                agg_slice = enc_vect
            else:
                agg_slice += enc_vect

        # need to create a tensor with the same shape as agg_slice
        # and fill it with 1.0 / N, where N is the number of slices
        # then divide agg_slice by this tensor
        scale_tensor = torch.ones(agg_slice.shape) / len(model_slices[model_slice])
        # scale_enc_tensor = ts.ckks_tensor(context, scale_tensor)
        scale_enc_tensor = scale_tensor
        agg_slice = agg_slice.mul(scale_enc_tensor)

        logger.info("Saving slice {}".format(model_slice))
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, model_slice), "wb") as f:
            f.write(agg_slice.serialize())


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

    # with open(args.context, "r", encoding="utf-8") as f:
    #     bytes_context = base64.b64decode(f.read())

    # ts_context = ts.context_from(bytes_context)

    # run(args.checkpoints, args.output, ts_context, slice_extension=args.extension)
    run(args.checkpoints, args.output)


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
