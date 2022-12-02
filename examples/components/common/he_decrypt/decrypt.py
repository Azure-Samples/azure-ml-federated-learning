"""Aggregate multiple pytorch models using FedAvg."""
import os
import argparse
import logging
import sys
import glob

import torch
import tenseal as ts


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
        "--model",
        type=str,
        required=True
    )
    parser.add_argument("--extension", type=str, default="pt", help="model extension")
    parser.add_argument(
        "--output", type=str, required=True
    )
    parser.add_argument(
        "--key_input", type=str, required=False, default=None, help="use this path for key"
    )

    return parser


class PyTorchStateDictHomomorphicEncryption:
    """Class to handle HE of pytorch models."""
    def __init__(self, context):
        """Constructor."""
        # generate HE context
        self.context = context
        # model state dict
        self.model_state_dict = None
        # if model is a class, we'll store it here
        self.model_object = None
        # keep track of the class and keys in the model (to test consistency)
        self.model_class = "NoneType"
        # a logger
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path: str):
        """Add one model to the average.

        Args:
            model_path (str): path to the model to add
        """
        self.model_state_dict = torch.load(model_path)
        self.model_class = self.model_state_dict.__class__.__name__

        if self.model_class != "OrderedDict":
            # if the model loaded is actually a class, we need to extract the state_dict
            self.model_object = self.model_state_dict
            self.model_state_dict = self.model_object.state_dict()

        self.ref_keys = set(self.model_state_dict.keys())

        self.logger.info(
            f"Loaded model from path={model_path}, class={self.model_class}, keys={self.ref_keys}"
        )

    def decrypt(self):
        for key in self.ref_keys:
            self.logger.info("Encrypting model key={}".format(key))
            self.model_state_dict[key] = ts.ckks_tensor(self.context, self.model_state_dict[key], batch=False)

    def save_model(self, model_path: str):
        """Save the averaged model.

        Args:
            model_path (str): path to save the model to
        """
        if self.model_class == "OrderedDict":
            self.logger.info(f"Saving state dict to path={model_path}")
            torch.save(self.model_state_dict, model_path)
        else:
            self.logger.info(f"Saving model object to path={model_path}")
            self.model_object.load_state_dict(self.model_state_dict)
            torch.save(self.model_object, model_path)

    def save_key(self, key_path:str):
        """Save the encryption key."""


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

    if args.key_input:
        # load the key from a file
        logging.getLogger(__name__).info("Reading key from path={}".format(args.key_input))
        with open(args.key_input, "rb") as f:
            data = f.read()
        ts_context = ts.enc_context.Context.load(data)
    else:
        # generate a new key
        ts_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=4096,
            plain_modulus=1032193
        )

        # bits_scale: controls precision of the fractional part
        bits_scale = 24
        # set the scale
        ts_context.global_scale = pow(2, bits_scale)
        # galois keys are required to do ciphertext rotations
        ts_context.generate_galois_keys()

    if args.key_output:
        logging.getLogger(__name__).info("Saving key to path={}".format(args.key_output))
        with open(os.path.join(args.key_output, "he_key.bin"), "wb") as out_file:
            out_file.write(ts_context.serialize(
                save_public_key=True,
                save_secret_key=True,
                save_galois_keys=True,
                save_relin_keys=True,
            ))

    model_handler = PyTorchStateDictHomomorphicEncryption(ts_context)

    if os.path.isdir(args.model):
        model_paths = glob.glob(os.path.join(args.model, f"*.{args.extension}"))
        assert len(model_paths) == 1, f"Found {len(model_paths)} models in {args.model}, expected one exactly."
        model_path = model_paths[0]
    else:
        model_path = args.model

    model_handler.load_model(model_path)
    #model_handler.encrypt()

    model_handler.save_model(os.path.join(args.output, f"he_model.{args.extension}"))


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
