import os
import argparse
import logging
import sys
import glob

import torch


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

    parser.add_argument("--checkpoints", type=str, required=True, nargs="+", help="list of paths or directories to search for model files")
    parser.add_argument("--extension", type=str, default="pt", help="model extension")
    parser.add_argument("--output", type=str, required=True, help="where to write the averaged model")

    return parser


class PyTorchStadeDictFedAvg:
    def __init__(self):
        self.model_class = "NoneType"
        self.model_count = 0
        self.model_type = "state_dict"
        self.model_object = None
        self.avg_state_dict = None
        self.ref_keys = {}

    def add_model(self, model_path: str):
        if self.avg_state_dict is None:
            # no model yet, nothing to average
            self.avg_state_dict = torch.load(model_path)
            self.model_class = self.avg_state_dict.__class__.__name__

            if self.model_class != "OrderedDict":
                # if the model loaded is actually a class, we need to extract the state_dict
                self.model_object = self.avg_state_dict
                self.avg_state_dict = self.model_object.state_dict()

            self.ref_keys = set(self.avg_state_dict.keys())

            print(f"Loaded model from path={model_path}, class={self.model_class}, keys={self.ref_keys}")
            self.model_count = 1

        else:
            # load the new model
            add_model = torch.load(model_path)
            assert add_model.__class__.__name__ == self.model_class, f"Model class mismatch: {add_model.__class__.__name__} != {self.model_class}"

            if self.model_class != "OrderedDict":
                # if the model loaded is actually a class, we need to extract the state_dict
                model_object = add_model
                add_model = model_object.state_dict()

            add_model_keys = set(add_model.keys())
            assert (
                self.ref_keys == add_model_keys
            ), f"model has keys {add_model_keys} != first model keys {self.ref_keys}"

            print(f"Loaded model from path={model_path}, class={self.model_class}, keys=IDEM")

            # rolling average
            for key in self.ref_keys:
                self.avg_state_dict[key] = torch.div(
                    self.avg_state_dict[key] * self.model_count + add_model[key],
                    float(self.model_count + 1),
                )

            self.model_count += 1

            # would this help free memory?
            del add_model

    def save_model(self, model_path: str):
        if self.model_class == "OrderedDict":
            torch.save(self.avg_state_dict, model_path)
        else:
            self.model_object.load_state_dict(self.avg_state_dict)
            torch.save(self.model_object, model_path)


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
            for f in glob.glob(os.path.join(model_path, f"*.{args.extension}"), recursive=True):
                model_paths.append(f)
        else:
            model_paths.append(model_path)

    model_handler = PyTorchStadeDictFedAvg()
    for model_path in model_paths:
        model_handler.add_model(model_path)

    model_handler.save_model(args.output)

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
