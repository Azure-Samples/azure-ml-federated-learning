"""Script for mock components."""
import argparse
import logging
import sys
import os
import copy

import mlflow
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.functional import precision_recall, accuracy, auroc
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
from typing import List
import models as models
import datasets as datasets
from distutils.util import strtobool

# DP
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# helper with confidentiality
from confidential_io import EncryptedFile


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
        train_data_dir="./",
        test_data_dir="./",
        model_path=None,
        lr=0.01,
        epochs=1,
        batch_size=10,
        dp=False,
        dp_target_epsilon=50.0,
        dp_target_delta=1e-5,
        dp_max_grad_norm=1.0,
        total_num_of_iterations=1,
        experiment_name="default-experiment",
        iteration_name="default-iteration",
        device_id=None,
        distributed=False,
    ):
        """Credit Card Fraud Trainer trains simple model on the Fraud dataset.

        Args:
            model_name(str): Name of the model to use for training, options: SimpleLinear, SimpleLSTM, SimpleVAE.
            train_data_dir(str, optional): Training data directory path.
            test_data_dir(str, optional): Testing data directory path.
            lr (float, optional): Learning rate. Defaults to 0.01.
            epochs (int, optional): Epochs. Defaults to 1.
            batch_size (int, optional): DataLoader batch size. Defaults to 64.
            dp (bool, optional): Differential Privacy. Default is False (Note: dp, dp_target_epsilon, dp_target_delta, dp_max_grad_norm, and total_num_of_iterations are defined for the only purpose of DP and can be ignored when users don't want to use Differential Privacy)
            dp_target_epsilon (float, optional): DP target epsilon. Default is 50.0
            dp_target_delta (float, optional): DP target delta. Default is 1e-5
            dp_max_grad_norm (float, optional): DP max gradient norm. Default is 1.0
            total_num_of_iterations (int, optional): Total number of iterations. Defaults to 1
            device_id (int, optional): Device id to run training on. Default to None.
            distributed (bool, optional): Whether to run distributed training. Default to False.

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
        self._distributed = distributed

        self.device_ = (
            torch.device(
                torch.device("cuda", device_id) if torch.cuda.is_available() else "cpu"
            )
            if device_id is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device_}")

        if self._distributed:
            self._rank = device_id
            logger.info(f"Rank: {self._rank}")
        else:
            self._rank = None

        self.train_dataset_, self.test_dataset_, self._input_dim = self.load_dataset(
            train_data_dir, test_data_dir, model_name
        )

        if self._distributed:
            logger.info("Setting up distributed samplers.")
            self.train_sampler_ = DistributedSampler(self.train_dataset_)
            self.test_sampler_ = DistributedSampler(self.test_dataset_)
        else:
            self.train_sampler_ = None
            self.test_sampler_ = None

        # get number of cpu to load data for each gpu
        num_workers_per_gpu = int(
            multiprocessing.cpu_count() // int(os.environ.get("WORLD_SIZE", "1"))
        )
        logger.info(f"The num_work per GPU is: {num_workers_per_gpu}")

        self.train_loader_ = DataLoader(
            self.train_dataset_,
            batch_size=batch_size,
            num_workers=num_workers_per_gpu,
            shuffle=(self.train_sampler_ is None),
            prefetch_factor=3,
            sampler=self.train_sampler_,
        )
        self.test_loader_ = DataLoader(
            self.test_dataset_,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.test_sampler_,
        )

        logger.info(f"Train loader steps: {len(self.train_loader_)}")
        logger.info(f"Test loader steps: {len(self.test_loader_)}")

        # Build model
        self.model_ = getattr(models, model_name)(self._input_dim).to(self.device_)
        if self._distributed:
            self.model_ = DDP(
                self.model_,
                device_ids=[self._rank] if self._rank is not None else None,
                output_device=self._rank,
            )
        self._model_path = model_path

        self.criterion_ = nn.BCELoss()

        # DP
        logger.info(f"DP: {dp}")
        if dp:
            if not ModuleValidator.is_valid(self.model_):
                self.model_ = ModuleValidator.fix(self.model_)

        self.optimizer_ = Adam(self.model_.parameters(), lr=self._lr, weight_decay=1e-5)

        if dp:
            privacy_engine = PrivacyEngine(secure_mode=False)
            """secure_mode: Set to True if cryptographically strong DP guarantee is
            required. secure_mode=True uses secure random number generator for
            noise and shuffling (as opposed to pseudo-rng in vanilla PyTorch) and
            prevents certain floating-point arithmetic-based attacks.
            See :meth:~opacus.optimizers.optimizer._generate_noise for details.
            When set to True requires torchcsprng to be installed"""
            (
                self.model_,
                self.optimizer_,
                self.train_loader_,
            ) = privacy_engine.make_private_with_epsilon(
                module=self.model_,
                optimizer=self.optimizer_,
                data_loader=self.train_loader_,
                epochs=total_num_of_iterations * epochs,
                target_epsilon=dp_target_epsilon,
                target_delta=dp_target_delta,
                max_grad_norm=dp_max_grad_norm,
            )

            """
            You can also obtain their counterparts by passing the noise multiplier. 
            Please refer to the following function.
            privacy_engine.make_private(
                module=self.model_,
                optimizer=self.optimizer_,
                data_loader=self.train_loader_,
                noise_multiplier=dp_noise_multiplier,
                max_grad_norm=dp_max_grad_norm,
            )
            """
            logger.info(
                f"Target epsilon: {dp_target_epsilon}, delta: {dp_target_delta} and noise multiplier: {self.optimizer_.noise_multiplier}"
            )

    def load_dataset(self, train_data_dir, test_data_dir, model_name):
        """Load dataset from {train_data_dir} and {test_data_dir}

        Args:
            train_data_dir(str): Training data directory path
            test_data_dir(str): Testing data directory path
            model_name(str): Name of the model to use
        """
        logger.info(f"Train data dir: {train_data_dir}, Test data dir: {test_data_dir}")
        with EncryptedFile(train_data_dir + "/fraud_weight.txt") as f:
            self.fraud_weight_ = np.loadtxt(f).item()
        with EncryptedFile(train_data_dir + "/data.csv") as f:
            train_df = pd.read_csv(f)
        with EncryptedFile(test_data_dir + "/data.csv") as f:
            test_df = pd.read_csv(f)
        if model_name == "SimpleLinear":
            train_dataset = datasets.FraudDataset(train_df)
            test_dataset = datasets.FraudDataset(test_df)
        else:
            train_dataset = datasets.FraudTimeDataset(train_df)
            test_dataset = datasets.FraudTimeDataset(test_df)

        logger.info(
            f"Train data samples: {len(train_df)}, Test data samples: {len(test_df)}"
        )

        return train_dataset, test_dataset, train_df.shape[1] - 1

    def log_params(self, client, run_id):
        if run_id:
            client.log_param(
                run_id=run_id,
                key=f"learning_rate {self._experiment_name}",
                value=self._lr,
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
        if run_id:
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
            if self._distributed:
                # DDP comes with "module." prefix: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
                self.model_.module.load_state_dict(
                    torch.load(checkpoint + "/model.pt", map_location=self.device_)
                )
            else:
                self.model_.load_state_dict(
                    torch.load(checkpoint + "/model.pt", map_location=self.device_)
                )

        with mlflow.start_run() as mlflow_run:
            num_of_batches_before_logging = 5

            # get Mlflow client and root run id
            mlflow_client = mlflow.tracking.client.MlflowClient()
            root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
            logger.debug(f"Root runId: {root_run_id}")

            # log params
            self.log_params(mlflow_client, root_run_id)

            logger.debug("Local training started")

            train_metrics = RunningMetrics(
                ["loss", "accuracy", "precision", "recall", "auroc"], prefix="train"
            )

            for epoch in range(1, self._epochs + 1):
                self.model_.train()
                train_metrics.reset_global()

                for i, batch in enumerate(self.train_loader_):
                    data, labels = batch[0].to(self.device_), batch[1].to(self.device_)
                    # Zero gradients for every batch
                    self.optimizer_.zero_grad()

                    predictions, net_loss = self.model_(data)
                    self.criterion_.weight = (
                        ((labels == 1) * (self.fraud_weight_ - 1)) + 1
                    ).to(self.device_)
                    # Compute loss
                    loss = self.criterion_(predictions, labels.type(torch.float))
                    if net_loss is not None:
                        loss += net_loss * 1e-5

                    # Compute gradients and adjust learning weights
                    loss.backward()
                    self.optimizer_.step()

                    precision, recall = precision_recall(
                        preds=predictions.detach(), target=labels
                    )
                    auroc_metric = auroc(
                        preds=predictions.detach(), target=labels, task="binary"
                    )
                    train_metrics.add_metric("auroc", auroc_metric.item())
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

                    if (i + 1) % num_of_batches_before_logging == 0 or (i + 1) == len(
                        self.train_loader_
                    ):
                        log_message = [
                            f"Epoch: {epoch}/{self._epochs}",
                            f"Iteration: {i+1}/{len(self.train_loader_)}",
                        ]

                        for name, value in train_metrics.get_step().items():
                            log_message.append(f"{name}: {value}")
                            self.log_metrics(
                                mlflow_client,
                                root_run_id,
                                name,
                                value,
                            )
                        logger.info(", ".join(log_message))
                        train_metrics.reset_step()

                test_metrics = self.test()

                log_message = [
                    f"Epoch: {epoch}/{self._epochs}",
                ]

                # log test metrics after each epoch
                for name, value in train_metrics.get_global().items():
                    log_message.append(f"{name}: {value}")
                    self.log_metrics(
                        mlflow_client,
                        root_run_id,
                        name,
                        value,
                    )
                for name, value in test_metrics.get_global().items():
                    log_message.append(f"{name}: {value}")
                    self.log_metrics(
                        mlflow_client,
                        root_run_id,
                        name,
                        value,
                    )
                logger.info(", ".join(log_message))

            log_message = [
                f"End of training",
            ]
            # log metrics at the pipeline level
            for name, value in train_metrics.get_global().items():
                log_message.append(f"{name}: {value}")
                if not self._distributed or self._rank == 0:
                    self.log_metrics(
                        mlflow_client,
                        root_run_id,
                        name,
                        value,
                        pipeline_level=True,
                    )

            for name, value in test_metrics.get_global().items():
                log_message.append(f"{name}: {value}")
                if not self._distributed or self._rank == 0:
                    self.log_metrics(
                        mlflow_client,
                        root_run_id,
                        name,
                        value,
                        pipeline_level=True,
                    )
            logger.info(", ".join(log_message))

    def test(self):
        """Test the trained model and report test loss and accuracy"""
        test_metrics = RunningMetrics(
            ["loss", "accuracy", "precision", "recall", "auroc"], prefix="test"
        )

        self.model_.eval()
        with torch.no_grad():
            for data, target in self.test_loader_:
                data, labels = data.to(self.device_), target.to(self.device_)
                predictions, net_loss = self.model_(data)

                self.criterion_.weight = (
                    ((labels == 1) * (self.fraud_weight_ - 1)) + 1
                ).to(self.device_)
                loss = self.criterion_(predictions, labels.type(torch.float)).item()
                if net_loss is not None:
                    loss += net_loss * 1e-5

                precision, recall = precision_recall(
                    preds=predictions.detach(), target=labels
                )
                auroc_metric = auroc(
                    preds=predictions.detach(), target=labels, task="binary"
                )
                test_metrics.add_metric("auroc", auroc_metric.item())
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

        if not self._distributed:
            logger.debug("Save model")
            torch.save(self.model_.state_dict(), self._model_path)
            logger.info(f"Model saved to {self._model_path}")
        elif self._rank == 0:
            # DDP comes with "module." prefix: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
            logger.debug("Save model")
            torch.save(self.model_.module.state_dict(), self._model_path)
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
    parser.add_argument(
        "--dp", type=strtobool, required=False, help="differential privacy"
    )
    parser.add_argument(
        "--dp_target_epsilon", type=float, required=False, help="DP target epsilon"
    )
    parser.add_argument(
        "--dp_target_delta", type=float, required=False, help="DP target delta"
    )
    parser.add_argument(
        "--dp_max_grad_norm", type=float, required=False, help="DP max gradient norm"
    )
    parser.add_argument(
        "--total_num_of_iterations",
        type=int,
        required=False,
        help="Total number of iterations",
    )
    return parser


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    logger.info(f"Distributed process rank: {os.environ['RANK']}")
    logger.info(f"Distributed world size: {os.environ['WORLD_SIZE']}")

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and torch.cuda.is_available():
        dist.init_process_group(
            "nccl",
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
    elif int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("gloo")

    trainer = CCFraudTrainer(
        model_name=args.model_name,
        train_data_dir=args.train_data,
        test_data_dir=args.test_data,
        model_path=args.model_path + "/model.pt",
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dp=args.dp,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_target_delta=args.dp_target_delta,
        dp_max_grad_norm=args.dp_max_grad_norm,
        total_num_of_iterations=args.total_num_of_iterations,
        experiment_name=args.metrics_prefix,
        iteration_name=args.iteration_name,
        device_id=int(os.environ.get("LOCAL_RANK", "0")),
        distributed=int(os.environ.get("WORLD_SIZE", "1")) > 1
        and torch.cuda.is_available(),
    )
    trainer.execute(args.checkpoint)

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.destroy_process_group()


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

    logger.info(f"Running script with arguments: {args}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA devices count: {torch.cuda.device_count()}")

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
