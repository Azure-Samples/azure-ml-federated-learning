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
    return global_model


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """
    logger.debug("Get client models")
    client_models = get_client_models(args)
    logger.info(f"Total number of client models: {len(client_models)}")

    logger.debug(f"Get global model")
    global_model = get_global_model(args)

    logger.debug("aggregate model weights")
    global_model = aggregate_model_weights(global_model, client_models)

    logger.info("Saving model weights")
    torch.save(global_model.state_dict(), args.aggregated_output + "/model.pt")


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
    run(args)


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
