"""Script for mock components."""
import argparse
import logging
import sys
import copy
import os
from aml_comm import AMLCommSocket

import mlflow
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from typing import List
import models as models
import datasets as datasets


class RunningMetrics:
    def __init__(self, metrics: List[str], prefix: str = None) -> None:
        self._allowed_metrics = copy.deepcopy(metrics)
        self._running_step_metrics = {name: 0 for name in metrics}
        self._running_global_metrics = {name: 0 for name in metrics}
        self._batch_count_step = 0
        self._batch_count_global = 0
        self._prefix = "" if prefix is None else prefix

    def add_metric(self, name: str, value: float):
        """Add measurement to the set

        Args:
            name: name of the metrics
            value: value of the metrics
        """
        if name not in self._allowed_metrics:
            raise ValueError(f"Metric with name '{name}' not in logged metrics")
        self._running_step_metrics[name] += value
        self._running_global_metrics[name] += value

    def step(self):
        """Increases number of measurements taken. Must be called after every batch"""
        self._batch_count_step += 1
        self._batch_count_global += 1

    def reset_step(self):
        """Soft reset of the counter. Only reset counter for subset of batches."""
        self._batch_count_step = 0
        self._running_step_metrics = {name: 0 for name in self._running_step_metrics}

    def reset_global(self):
        """Reset all counters and steps"""
        self.reset_step()
        self._batch_count_global = 0
        self._running_global_metrics = {
            name: 0 for name in self._running_global_metrics
        }

    def get_step(self):
        """Provide average value for every metric since last `reset_step` call"""
        return {
            f"{self._prefix}_{name}": value / self._batch_count_step
            for name, value in self._running_step_metrics.items()
        }

    def get_global(self):
        """Provide average value for every metric since last `reset_global` call"""
        return {
            f"{self._prefix}_{name}": value / self._batch_count_global
            for name, value in self._running_global_metrics.items()
        }


class CCFraudTrainer:
    def __init__(
        self,
        model_name,
        global_rank,
        global_size,
        global_comm,
        train_data_dir="./",
        test_data_dir="./",
        model_path=None,
        lr=0.01,
        epochs=1,
        batch_size=10,
        experiment_name="default-experiment",
        iteration_name="default-iteration",
    ):
        """Credit Card Fraud Trainer trains simple model on the Fraud dataset.

        Args:
            model_name(str): Name of the model to use for training, options: SimpleLinear, SimpleLSTM, SimpleVAE
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 1
            batch_size (int, optional): DataLoader batch size. Defaults to 64.

        Attributes:
            model_: Model
            criterion_: BCELoss loss
            optimizer_: Stochastic gradient descent
            train_dataset_: Training Dataset obj
            train_loader_: Training DataLoader
            test_dataset_: Testing Dataset obj
            test_loader_: Testing DataLoader
        """

        # Training setup
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._experiment_name = experiment_name
        self._iteration_name = iteration_name
        self._global_rank = global_rank
        self._global_size = global_size
        self._global_comm = global_comm

        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device_}")

        self.train_dataset_, self.test_dataset_, self._input_dim = self.load_dataset(
            train_data_dir, test_data_dir, model_name
        )
        self.train_loader_ = DataLoader(
            self.train_dataset_, batch_size=batch_size, shuffle=False
        )
        self.test_loader_ = DataLoader(
            self.test_dataset_, batch_size=batch_size, shuffle=False
        )

        # Build model
        self._model_name = model_name
        if self._global_rank == 0:
            self.model_ = getattr(models, model_name + "Top")().to(self.device_)
        else:
            self.model_ = getattr(models, model_name + "Bottom")(self._input_dim).to(
                self.device_
            )
        self._model_path = model_path

        self.criterion_ = nn.BCELoss()
        self.optimizer_ = SGD(self.model_.parameters(), lr=self._lr, weight_decay=1e-5)

    def load_dataset(self, train_data_dir, test_data_dir, model_name):
        """Load dataset from {train_data_dir} and {test_data_dir}

        Args:
            train_data_dir(str): Training data directory path
            test_data_dir(str): Testing data directory path
            model_name(str): Name of the model to use
        """
        logger.info(f"Train data dir: {train_data_dir}, Test data dir: {test_data_dir}")
        if self._global_rank == 0:
            self.fraud_weight_ = np.loadtxt(train_data_dir + "/fraud_weight.txt").item()
        train_df = pd.read_csv(train_data_dir + "/data.csv", index_col=0)
        test_df = pd.read_csv(test_data_dir + "/data.csv", index_col=0)
        if model_name == "SimpleLinear":
            train_dataset = datasets.FraudDataset(train_df)
            test_dataset = datasets.FraudDataset(test_df)
        else:
            train_dataset = datasets.FraudTimeDataset(train_df)
            test_dataset = datasets.FraudTimeDataset(test_df)

        logger.info(
            f"Train data samples: {len(train_df)}, Test data samples: {len(test_df)}"
        )

        return train_dataset, test_dataset, train_dataset.features_count()

    def log_params(self, client, run_id):
        client.log_param(
            run_id=run_id, key=f"learning_rate {self._experiment_name}", value=self._lr
        )
        client.log_param(
            run_id=run_id, key=f"epochs {self._experiment_name}", value=self._epochs
        )
        client.log_param(
            run_id=run_id,
            key=f"batch_size {self._experiment_name}",
            value=self._batch_size,
        )
        client.log_param(
            run_id=run_id,
            key=f"loss {self._experiment_name}",
            value=self.criterion_.__class__.__name__,
        )
        client.log_param(
            run_id=run_id,
            key=f"optimizer {self._experiment_name}",
            value=self.optimizer_.__class__.__name__,
        )

    def log_metrics(self, client, run_id, key, value, pipeline_level=False):
        if pipeline_level:
            client.log_metric(
                run_id=run_id,
                key=f"{self._experiment_name}/{key}",
                value=value,
            )
        else:
            client.log_metric(
                run_id=run_id,
                key=f"{self._iteration_name}/{self._experiment_name}/{key}",
                value=value,
            )

    def local_train(self, checkpoint):
        """Perform local training for a given number of epochs

        Args:
            checkpoint: Previous model checkpoint from where training has to be started.
        """

        if checkpoint:
            self.model_.load_state_dict(torch.load(checkpoint + "/model.pt"))

        with mlflow.start_run() as mlflow_run:
            # get Mlflow client and root run id
            mlflow_client = mlflow.tracking.client.MlflowClient()
            root_run_id = os.environ.get("AZUREML_ROOT_RUN_ID")
            logger.debug(f"Root runId: {root_run_id}")

            # log params
            self.log_params(mlflow_client, root_run_id)

            logger.debug("Local training started")

            train_metrics = RunningMetrics(
                ["loss", "accuracy", "precision", "recall"], prefix="train"
            )

            for _ in range(1, self._epochs + 1):
                self.model_.train()
                train_metrics.reset_global()

                for i, batch in enumerate(self.train_loader_):
                    data = batch.to(self.device_)
                    # Zero gradients for every batch
                    self.optimizer_.zero_grad()

                    output, net_loss = self.model_(data)
                    output = output.contiguous()
                    self._global_comm.send(output, 0)
                    if net_loss is not None:
                        net_loss += net_loss * 1e-5
                        net_loss.backward(retain_graph=True)
                    gradient = self._global_comm.recv(0).to(self.device_)
                    output.backward(gradient)
                    self.optimizer_.step()

                self.test()

            log_message = [
                f"End of training",
            ]
            logger.info(", ".join(log_message))

    def test(self):
        """Test the trained model and report test loss and accuracy"""
        self.model_.eval()

        with torch.no_grad():
            for batch in self.test_loader_:
                data = batch.to(self.device_)

                if self._global_rank != 0:
                    output = self.model_(data)[0].contiguous()
                    self._global_comm.send(output, 0)

    def execute(self, checkpoint=None):
        """Bundle steps to perform local training, model testing and finally saving the model.

        Args:
            checkpoint: Previous model checkpoint from where training has to be started.
        """
        logger.debug("Start training")
        self.local_train(checkpoint)

        logger.debug("Save model")
        torch.save(self.model_.state_dict(), self._model_path)
        logger.info(f"Model saved to {self._model_path}")


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

    parser.add_argument("--train_data", type=str, required=True, help="")
    parser.add_argument("--test_data", type=str, required=True, help="")
    parser.add_argument("--checkpoint", type=str, required=False, help="")
    parser.add_argument("--model_path", type=str, required=True, help="")
    parser.add_argument("--model_name", type=str, required=True, help="")
    parser.add_argument(
        "--global_size",
        type=int,
        required=True,
        help="Number of silos",
    )
    parser.add_argument(
        "--global_rank",
        type=int,
        required=True,
        help="Index of the current silo",
    )
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    parser.add_argument(
        "--iteration_name", type=str, required=False, help="Iteration name"
    )

    parser.add_argument(
        "--lr", type=float, required=False, help="Training algorithm's learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="Total number of epochs for local training",
    )
    parser.add_argument("--batch_size", type=int, required=False, help="Batch Size")
    return parser


def run(args, global_comm):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    trainer = CCFraudTrainer(
        model_name=args.model_name,
        train_data_dir=args.train_data,
        test_data_dir=args.test_data,
        model_path=args.model_path + "/model.pt",
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        experiment_name=args.metrics_prefix,
        iteration_name=args.iteration_name,
        global_rank=args.global_rank,
        global_size=args.global_size,
        global_comm=global_comm,
    )
    trainer.execute(args.checkpoint)


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

    global_comm = AMLCommSocket(
        args.global_rank, args.global_size, os.environ.get("AZUREML_ROOT_RUN_ID")
    )

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Running script with arguments: {args}")
    run(args, global_comm)

    # log messaging stats
    with mlflow.start_run() as mlflow_run:
        mlflow_client = mlflow.tracking.client.MlflowClient()
        global_comm.log_stats(mlflow_client)


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
