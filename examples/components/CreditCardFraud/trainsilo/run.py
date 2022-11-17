"""Script for mock components."""
import argparse
import logging
import sys

import mlflow
import torch
import pandas as pd
from torch import nn
from torchmetrics.functional import precision_recall, accuracy
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from mlflow import log_metric, log_param
from typing import List
import models as models


class RunningMetrics:
    def __init__(self, metrics: List[str], prefix: str = None) -> None:
        self._running_metrics = {name: 0 for name in metrics}
        self._batch_count = 0
        self._prefix = "" if prefix is None else prefix

    def add_metric(self, name: str, value: float):
        if name not in self._running_metrics:
            raise ValueError(f"Metric with name '{name}' not in logged metrics")
        self._running_metrics[name] += value

    def step(self):
        self._batch_count += 1

    def reset(self):
        self._batch_count = 0
        self._running_metrics = {name: 0 for name in self._running_metrics}

    def get_all(self):
        return {
            f"{self._prefix}_{name}": value / self._batch_count
            for name, value in self._running_metrics.items()
        }


class FraudDataset(Dataset):
    """FraudDataset Dataset - combination of features and labels

    Args:
        feature: Transaction detail tensors
        target: Tensor of labels corresponding to features

    Returns:
        None
    """

    def __init__(self, df):
        self.X = torch.tensor(
            df.loc[:, df.columns != "is_fraud"].values, dtype=torch.float
        )
        self.Y = torch.tensor(df.loc[:, "is_fraud"].values, dtype=torch.int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is None:
            return [self.X[idx]]
        return self.X[idx], self.Y[idx]


class CCFraudTrainer:
    def __init__(
        self,
        model_name,
        input_dim=576,
        train_data_dir="./",
        test_data_dir="./",
        model_path=None,
        lr=0.01,
        epochs=1,
        batch_size=10000,
        experiment_name="default-experiment",
        iteration_name="default-iteration",
    ):
        """Credit Card Fraud Trainer trains simple model on the Fraud dataset.

        Args:
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
        self._input_dim = input_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._experiment_name = experiment_name
        self._iteration_name = iteration_name

        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dataset_, self.test_dataset_ = self.load_dataset(
            train_data_dir, test_data_dir
        )
        self.train_loader_ = DataLoader(
            self.train_dataset_, batch_size=batch_size, shuffle=True
        )
        self.test_loader_ = DataLoader(
            self.test_dataset_, batch_size=batch_size, shuffle=True
        )

        # Build model
        self.model_ = getattr(models, model_name)(self.train_dataset_.X.shape[1]).to(
            self.device_
        )
        self._model_path = model_path

        self.criterion_ = nn.BCELoss()
        self.optimizer_ = Adam(self.model_.parameters(), lr=self._lr)

    def load_dataset(self, train_data_dir, test_data_dir):
        """Load dataset from {train_data_dir} and {test_data_dir}

        Args:
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
        """
        logger.info(f"Train data dir: {train_data_dir}, Test data dir: {test_data_dir}")
        train_df = pd.read_csv(train_data_dir + "/data.csv")
        test_df = pd.read_csv(test_data_dir + "/data.csv")
        train_dataset = FraudDataset(train_df)
        test_dataset = FraudDataset(test_df)

        logger.info(
            f"Train data samples: {len(train_df)}, Test data samples: {len(test_df)}"
        )

        self.fraud_weight_ = (
            train_df["is_fraud"].value_counts()[0]
            / train_df["is_fraud"].value_counts()[1]
        )

        return train_dataset, test_dataset

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
            num_of_batches_before_logging = 10

            # get Mlflow client and root run id
            mlflow_client = mlflow.tracking.client.MlflowClient()
            logger.debug(f"Root runId: {mlflow_run.data.tags.get('mlflow.rootRunId')}")
            root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")

            # log params
            self.log_params(mlflow_client, root_run_id)

            logger.debug("Local training started")

            train_metrics = RunningMetrics(
                ["loss", "accuracy", "precision", "recall"], prefix="train"
            )
            test_metrics = RunningMetrics(
                ["accuracy", "precision", "recall"], prefix="test"
            )
            for epoch in range(self._epochs):
                self.model_.train()

                for i, batch in enumerate(self.train_loader_):
                    data, labels = batch[0].to(self.device_), batch[1].to(self.device_)
                    # Zero gradients for every batch
                    self.optimizer_.zero_grad()

                    predictions = self.model_(data)
                    self.criterion_.weight = (
                        ((labels == 1) * (self.fraud_weight_ - 1)) + 1
                    ).to(self.device_)
                    # Compute loss
                    loss = self.criterion_(
                        predictions.reshape(-1), labels.type(torch.float)
                    )

                    # Compute gradients and adjust learning weights
                    loss.backward()
                    self.optimizer_.step()

                    precision, recall = precision_recall(
                        preds=predictions.detach(), target=labels
                    )
                    train_metrics.add_metric("precision", precision.item())
                    train_metrics.add_metric("recall", recall.item())
                    train_metrics.add_metric(
                        "accuracy",
                        accuracy(
                            preds=predictions.detach(), target=labels, threshold=0.5
                        ).item(),
                    )
                    train_metrics.add_metric(
                        "loss", (loss.detach() / data.shape[0]).item()
                    )
                    train_metrics.step()

                    if (i + 1) != 0 and (i + 1) % num_of_batches_before_logging == 0:
                        log_message = [
                            f"Epoch: {epoch}/{self._epochs}",
                            f"Iteration: {i+1}/{len(self.train_loader_)}",
                        ]

                        for name, value in train_metrics.get_all().items():
                            log_message.append(f"{name}: {value}")
                            self.log_metrics(
                                mlflow_client,
                                root_run_id,
                                name,
                                value,
                            )
                        logger.info(", ".join(log_message))

                test_metrics = self.test()

                log_message = [
                    f"Epoch: {epoch}/{self._epochs}",
                ]

                # log test metrics after each epoch
                for name, value in test_metrics.get_all().items():
                    log_message.append(f"{name}: {value}")
                    self.log_metrics(
                        mlflow_client,
                        root_run_id,
                        name,
                        value,
                    )
                logger.info(", ".join(log_message))

            # log metrics at the pipeline level
            for name, value in train_metrics.get_all().items():
                log_message.append(f"{name}: {value}")
                self.log_metrics(
                    mlflow_client,
                    root_run_id,
                    name,
                    value,
                    pipeline_level=True,
                )

            train_metrics.reset()

    def test(self):
        """Test the trained model and report test loss and accuracy"""
        test_metrics = RunningMetrics(
            ["loss", "accuracy", "precision", "recall"], prefix="test"
        )

        self.model_.eval()
        with torch.no_grad():
            for data, target in self.test_loader_:
                data, labels = data.to(self.device_), target.to(self.device_)
                predictions = self.model_(data)

                self.criterion_.weight = (
                    ((labels == 1) * (self.fraud_weight_ - 1)) + 1
                ).to(self.device_)
                loss = self.criterion_(
                    predictions.reshape(-1), labels.type(torch.float)
                ).item()

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

        return test_metrics

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


def run(args):
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
        experiment_name=args.metrics_prefix,
        iteration_name=args.iteration_name,
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
