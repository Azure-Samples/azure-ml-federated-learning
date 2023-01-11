"""Aggregate multiple pytorch models using FedAvg."""
import os
import argparse
import logging
import sys
import glob
import shutil
from distutils.util import strtobool
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        help="list of paths or directories to search for model files",
    )
    parser.add_argument("--extension", type=str, default="pt", help="model extension")
    parser.add_argument(
        "--output", type=str, required=True, help="where to write the averaged model"
    )
    parser.add_argument(
        "--ancillary_files",
        type=strtobool,
        default=False,
        help="Whether ancillary files need to be copied",
    )
    parser.add_argument(
        "--out_checkpoint_name",
        type=str,
        default="model",
        help=" the name of the output checkpoint, e.g. model for CCFRAUD/MNIST/NER/PNEUMONIA, finetuned_state_dict for Babel models",
    )

    return parser


class PyTorchStateDictFedAvg:
    """Class to handle FedAvg of pytorch models."""

    def __init__(self):
        """Constructor."""
        # below we keep the average of the models
        self.avg_state_dict = None
        # keep count of how many models were averaged (rolling)
        self.model_count = 0
        # if model is a class, we'll store it here
        self.model_object = None
        # keep track of the class and keys in the model (to test consistency)
        self.model_class = "NoneType"
        self.ref_keys = {}
        # a logger
        self.logger = logging.getLogger(__name__)

    def add_model(self, model_path: str):
        """Add one model to the average.
        Args:
            model_path (str): path to the model to add
        """
        if self.avg_state_dict is None:
            # no model yet, nothing to average
            self.avg_state_dict = torch.load(model_path, map_location=device)
            self.model_class = self.avg_state_dict.__class__.__name__

            if self.model_class != "OrderedDict":
                # if the model loaded is actually a class, we need to extract the state_dict
                self.model_object = self.avg_state_dict
                self.avg_state_dict = self.model_object.state_dict()

            self.ref_keys = set(self.avg_state_dict.keys())

            self.logger.info(
                f"Loaded model from path={model_path}, class={self.model_class}, keys={self.ref_keys}"
            )
            self.model_count = 1

        else:
            # load the new model
            model_to_add = torch.load(model_path, map_location=device)
            assert (
                model_to_add.__class__.__name__ == self.model_class
            ), f"Model class mismatch: {model_to_add.__class__.__name__} != {self.model_class}"

            if self.model_class != "OrderedDict":
                # if the model loaded is actually a class, we need to extract the state_dict
                model_object = model_to_add
                model_to_add = model_object.state_dict()

            model_to_add_keys = set(model_to_add.keys())
            assert (
                self.ref_keys == model_to_add_keys
            ), f"model has keys {model_to_add_keys} != first model keys {self.ref_keys}"

            self.logger.info(
                f"Loaded model from path={model_path}, class={self.model_class}, keys=IDEM"
            )

            # rolling average
            for key in self.ref_keys:
                self.avg_state_dict[key] = torch.div(
                    self.avg_state_dict[key] * self.model_count + model_to_add[key],
                    float(self.model_count + 1),
                )

            self.model_count += 1

            # would this help free memory?
            del model_to_add

    def save_model(self, model_path: str):
        """Save the averaged model.
        Args:
            model_path (str): path to save the model to
        """
        if self.model_class == "OrderedDict":
            self.logger.info(f"Saving state dict to path={model_path}")
            torch.save(self.avg_state_dict, model_path)
        else:
            self.logger.info(f"Saving model object to path={model_path}")
            if not (self.model_object is None):
                self.model_object.load_state_dict(self.avg_state_dict)
                torch.save(self.model_object, model_path)
            else:  # for dummy HELLOWORLD example, just write a simple text file
                with open(model_path, "w") as f:
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

    model_paths = []
    for model_path in args.checkpoints:
        if os.path.isdir(model_path):
            for f in glob.glob(
                os.path.join(model_path, f"*.{args.extension}"), recursive=True
            ):
                model_paths.append(f)
        else:
            model_paths.append(model_path)

    model_handler = PyTorchStateDictFedAvg()
    for model_path in model_paths:
        model_handler.add_model(model_path)

    # check if meta data needed, copy the whole dir
    if args.ancillary_files:
        for src in os.listdir(args.checkpoints[0]):
            src_path = os.path.join(args.checkpoints[0], src)
            if os.path.isfile(src_path):
                shutil.copy(src_path, f"{args.output}/{src}")
            else:
                shutil.copytree(src_path, f"{args.output}/{src}")

        # remove the checkpoint, it's from single silo
        out_checkpoint_pth = os.path.join(
            args.output, f"{args.out_checkpoint_name}.{args.extension}"
        )
        os.remove(out_checkpoint_pth)

    model_handler.save_model(
        os.path.join(args.output, f"{args.out_checkpoint_name}.{args.extension}")
    )


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
