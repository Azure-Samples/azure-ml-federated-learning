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

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--context", type=str, required=True)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path: str):
        """Load a pytorch model or state dict.

        Args:
            model_path (str): path to the model to add
        """
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")

        # no model yet, nothing to average
        self.model_state_dict = torch.load(model_path, map_location=self.device)
        self.model_class = self.model_state_dict.__class__.__name__

        if self.model_class != "OrderedDict":
            # if the model loaded is actually a class, we need to extract the state_dict
            self.model_object = self.model_state_dict
            self.model_state_dict = self.model_object.state_dict()

        self.ref_keys = set(self.model_state_dict.keys())

        self.logger.info(
            f"Loaded model from path={model_path}, class={self.model_class}, keys={self.ref_keys}"
        )

    def encrypt(self, output_path: str):
        # create the directory just in case
        os.makedirs(output_path, exist_ok=True)

        for key in sorted(self.ref_keys):
            self.logger.info(
                "Encrypting model key={} shape={}".format(
                    key, self.model_state_dict[key].shape
                )
            )

            if self.model_state_dict[key].shape == torch.Size([]):
                self.logger.info(
                    f"Skipping encrypting scalar {self.model_state_dict[key].numpy()}"
                )
                continue

            self.logger.info("... creating CKKS tensor ...")
            # ckks_row = ts.ckks_tensor(
            ckks_row = ts.ckks_vector(
                self.context, self.model_state_dict[key].numpy() #, batch=True
            )
            # ckks_row = ts.ckks_vector(self.context, list(self.model_state_dict[key].numpy()))

            self.logger.info("... serializing CKKS tensor ...")
            ckks_serialized = ckks_row.serialize()

            # save the encrypted tensor
            self.logger.info("... saving CKKS tensor ...")
            with open(os.path.join(output_path, key + ".bin"), "wb") as f:
                f.write(ckks_serialized)


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

    with open(args.context, "r", encoding="utf-8") as f:
        bytes_context = base64.b64decode(f.read())

    ts_context = ts.context_from(bytes_context)

    he_handler = PyTorchStateDictHomomorphicEncryption(ts_context)
    he_handler.load_model(args.input)
    he_handler.encrypt(args.output)


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
