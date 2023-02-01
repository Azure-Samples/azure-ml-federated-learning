"""Script for mock components."""
import argparse
import logging
import sys
import copy
import os

import mlflow
import torch
import pandas as pd
import numpy as np
from torch import nn
from torchmetrics.functional import precision_recall, accuracy
from torch.utils.data.dataloader import DataLoader
from mlflow import log_metric, log_param
from typing import List


TRAIN_COMPONENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../traininsilo"))
sys.path.insert(0, TRAIN_COMPONENT)
from run import RunningMetrics
import models as models
import datasets as datasets



def load_dataset( test_data_dir, model_name):
    """Load dataset from {test_data_dir}

    Args:
        test_data_dir(str): Testing data directory path
        model_name(str): Name of the model to use
    """
    logger.info(f" Test data dir: {test_data_dir}")
    test_df = pd.read_csv(test_data_dir + "/unfiltered_data.csv")
    if model_name == "SimpleLinear":
        test_dataset = datasets.FraudDataset(test_df)
    else:
        test_dataset = datasets.FraudTimeDataset(test_df)

    logger.info(
        f"Test data samples: {len(test_df)}"
    )

    return  test_dataset, test_df.shape[1] - 1


def log_metrics(client, run_id, key, value, experiment_name="default-experiment"):

    client.log_metric(
        run_id=run_id,
        key=f"{experiment_name}/{key}",
        value=value,
    )



def test(args):

    """Test the trained model and report test loss and accuracy"""
    test_metrics = RunningMetrics(
        ["loss", "accuracy", "precision", "recall"], prefix="test"
    )
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset, input_dim = load_dataset(args.test_data_dir, args.model_name)
    test_dataloader =  DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True
    )
    fraud_weight = np.loadtxt(args.fraud_weight_path + "/unfiltered_fraud_weight.txt").item()
    model_path = os.path.join(args.checkpoint, "model.pt")
    model = getattr(models, args.model_name)(input_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.BCELoss()
    all_predictions = None
    with mlflow.start_run() as mlflow_run:
        mlflow_client = mlflow.tracking.client.MlflowClient()
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        with torch.no_grad():
            for i, (data, target) in enumerate(test_dataloader):
                data, labels = data.to(device), target.to(device)
                predictions, net_loss = model(data)
                if i == 0:
                    all_predictions = predictions
                else:
                    np.concatenate((all_predictions, predictions), axis=0)
                criterion.weight = (((labels == 1) * (fraud_weight - 1)) + 1).to(device)
                loss = criterion(predictions, labels.type(torch.float)).item()
                if net_loss is not None:
                    loss += net_loss * 1e-5

                precision, recall = precision_recall(
                    preds=predictions.detach(), target=labels
                )
                test_metrics.add_metric("precision", precision.item())
                test_metrics.add_metric("recall", recall.item())
                test_metrics.add_metric(
                    "accuracy",
                    accuracy(preds=predictions.detach(), target=labels).item(),
                )
                test_metrics.add_metric("loss", loss / data.shape[0])
                test_metrics.step()

        np.savetxt(f"{args.predictions_path}/predictions.txt", all_predictions)
        log_message = []
        for name, value in test_metrics.get_global().items():
            log_message.append(f"{name}: {value}")
            log_metrics(
                mlflow_client,
                root_run_id,
                name,
                value
            )
        logger.info(", ".join(log_message))



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

    parser.add_argument("--test_data_dir", type=str, required=True, help="")
    parser.add_argument("--checkpoint", type=str, required=False, help="")

    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    parser.add_argument("--batch_size", type=int, required=False, help="Batch Size")

    parser.add_argument("--model_name", type=str, required=True, help="")
    parser.add_argument("--fraud_weight_path", type=str, required=True, help="")
    parser.add_argument("--predictions_path", type=str, required=True, help="")
    return parser



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
    test(args)


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
