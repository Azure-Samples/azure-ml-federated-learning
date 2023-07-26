"""Script for mock components."""
import argparse
import logging
import sys
import os
from distutils.util import strtobool
from aml_comm import AMLCommSocket, AMLCommRedis
from aml_smpc import AMLSMPC
from samplers import VerticallyDistributedBatchSampler

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


class BankMarketingTrainer:
    def __init__(
        self,
        model_name,
        global_rank,
        global_size,
        global_comm,
        data_dir="./",
        lr=0.01,
        epochs=1,
        batch_size=10,
        experiment_name="default-experiment",
    ):
        """Bank Marketing Trainer trains simple model on the Bank Marketing dataset.

        Args:
             model_name(str): Name of the model to use for training, options: SimpleLinear, SimpleLSTM, SimpleVAE.
             global_rank(int): Rank of the current node.
             global_size(int): Total number of nodes.
             global_comm(AMLComm): Communication method.
             data_dir(str, optional): Data directory path.
             batch_size (int, optional): DataLoader batch size. Defaults to 64.
             experiment_name (str, optional): Name of the experiment. Defaults to "default-experiment".

         Attributes:
             model_: Model
             device_: Location of the model
             dataset_: Dataset obj
             loader_: DataLoader obj
        """

        # Training setup
        self._batch_size = batch_size
        self._experiment_name = experiment_name
        self._global_rank = global_rank
        self._global_size = global_size
        self._global_comm = global_comm

        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device_}")

        self.dataset, self._input_dim = self.load_dataset(
            data_dir, model_name
        )
        self.sampler_ = VerticallyDistributedBatchSampler(
            data_source=self.dataset,
            batch_size=batch_size,
            comm=self._global_comm,
            rank=self._global_rank,
            world_size=self._global_size,
            shuffle=True,
        )
        
        self.loader_ = DataLoader(
            self.dataset, batch_sampler=self.sampler_
        )

        # Build model
        self._model_name = model_name
        self.model_ = getattr(models, model_name + "Bottom")(self._input_dim).to(
            self.device_
        )

    def load_dataset(self, data_dir, model_name):
        """Load dataset from {data_dir}

        Args:
            data_dir(str): Data directory path
            model_name(str): Name of the model to use
        """
        logger.info(f"Data dir: {data_dir}")
        df = pd.read_csv(data_dir + "/data.csv", index_col=0)
        dataset = datasets.BankMarketingDataset(df)

        logger.info(
            f"Data samples: {len(df)}"
        )

        return dataset, dataset.features_count()

    def log_params(self, client, run_id):
        client.log_param(
            run_id=run_id,
            key=f"batch_size {self._experiment_name}",
            value=self._batch_size,
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
                key=f"{self._experiment_name}/{key}",
                value=value,
            )

    def eval(self):
        """Test the trained model and report test loss and accuracy"""
        self.model_.eval()
        with torch.no_grad():
            for batch in self.loader_:
                data = batch.to(self.device_)

                # Send intermediate output to the global model
                # such that it can calculate metrics
                output = self.model_(data).contiguous()
                self._global_comm.send(output, 0)

    def execute(self, checkpoint=None):
        """Bundle steps to perform eval using checkpoint.

        Args:
            checkpoint: Previous model checkpoint from where training has to be started.
        """
        logger.debug("Start evaluation")
        assert checkpoint is not None, "Checkpoint path is required"
        self.model_.load_state_dict(torch.load(checkpoint + "/model.pt"))
        logger.debug("Checkpoint loaded")
        
        with mlflow.start_run() as mlflow_run:

            # get Mlflow client and root run id
            mlflow_client = mlflow.tracking.client.MlflowClient()
            root_run_id = os.environ.get("AZUREML_ROOT_RUN_ID")
            logger.debug(f"Root runId: {root_run_id}")

            # log params
            self.log_params(mlflow_client, root_run_id)

            logger.debug("Evaluation started")
            self.eval()


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

    parser.add_argument("--data", type=str, required=True, help="")
    parser.add_argument("--checkpoint", type=str, required=False, help="")
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
    parser.add_argument("--batch_size", type=int, required=False, help="Batch Size")
    parser.add_argument(
        "--communication_backend",
        type=str,
        required=False,
        default="socket",
        help="Type of communication to use between the nodes",
    )
    parser.add_argument(
        "--communication_encrypted",
        type=strtobool,
        required=False,
        default=False,
        help="Encrypt messages exchanged between the nodes",
    )
    return parser


def run(args, global_comm):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    trainer = BankMarketingTrainer(
        model_name=args.model_name,
        data_dir=args.data,
        batch_size=args.batch_size,
        experiment_name=args.metrics_prefix,
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

    if args.communication_encrypted:
        encryption = AMLSMPC()
    else:
        encryption = None

    if args.communication_backend == "socket":
        global_comm = AMLCommSocket(
            args.global_rank,
            args.global_size,
            os.environ.get("AZUREML_ROOT_RUN_ID"),
            encryption=encryption,
        )
    elif args.communication_backend == "redis":
        global_comm = AMLCommRedis(
            args.global_rank,
            args.global_size,
            os.environ.get("AZUREML_ROOT_RUN_ID"),
            encryption=encryption,
        )
    else:
        raise ValueError("Communication backend not supported")

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
