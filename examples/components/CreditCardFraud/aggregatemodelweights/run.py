"""Aggregate multiple pytorch models using FedAvg."""
import os
import argparse
import logging
import sys
import copy

import torch
import models as models


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

    parser.add_argument("--input_silo_1", type=str, required=True, help="")
    parser.add_argument("--input_silo_2", type=str, required=False, help="")
    parser.add_argument("--input_silo_3", type=str, required=False, help="")
    parser.add_argument("--aggregated_output", type=str, required=True, help="")
    parser.add_argument("--model_name", type=str, required=True, help="")
    return parser


def aggregate_model_weights(global_model, client_models):
    """
    This function has aggregation method 'mean'

    Args:
    client_models: list of client models
    """
    global_dict = copy.deepcopy(client_models[0].state_dict())

    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [
                client_models[i].state_dict()[k].float()
                for i in range(len(client_models))
            ],
            0,
        ).mean(0)
    global_model.load_state_dict(global_dict)

    return global_model


def get_model(model_path, model_name, input_dim=None):
    """Get the model having custom input dimensions.

    model_path: Pretrained model weights file path
    """

    if model_path:
        model_chkpt = torch.load(
            model_path + "/model.pt", map_location=torch.device("cpu")
        )
        input_dim = model_chkpt.input_dim

    assert input_dim is not None

    model = getattr(models, model_name)(input_dim)
    if model_path:
        model.load_state_dict(model_chkpt)
    return model


def get_client_models(args):
    """Get the list of client models.

    args: an argument parser instance
    """
    client_models = []
    for i in range(1, len(args.__dict__)):
        client_model_name = "input_silo_" + str(i)
        logger.debug(
            f"Collecting client model name: {client_model_name}, model path: {args.__dict__[client_model_name]}"
        )
        if client_model_name in args.__dict__:
            client_models.append(
                get_model(args.__dict__[client_model_name], args.model_name)
            )
    return client_models


def get_global_model(args):
    """Get the global model.

    args: an argument parser instance
    """

    global_model = get_model(
        args.aggregated_output
        if args.aggregated_output
        and os.path.isfile(args.aggregated_output + "/model.pt")
        else args.__dict__["input_silo_1"],
        args.model_name,
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
            self.avg_state_dict = torch.load(model_path)
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
            model_to_add = torch.load(model_path)
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
            for f in glob.glob(
                os.path.join(model_path, f"*.{args.extension}"), recursive=True
            ):
                model_paths.append(f)
        else:
            model_paths.append(model_path)

    model_handler = PyTorchStateDictFedAvg()
    for model_path in model_paths:
        model_handler.add_model(model_path)

    model_handler.save_model(os.path.join(args.output, f"model.{args.extension}"))


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
