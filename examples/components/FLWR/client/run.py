"""This script runs a Flower client inside an AzureML job.

The script will:
- fetch the server IP from a pipeline root tag using mlflow,
- add server name+ip to /etc/hosts to allow client code to find server,
- run the flower client setup.

NOTE: the script can take an input data --client_data folder
which AzureML will mount to the job.
"""

import argparse
import logging
import sys
import os.path
import time
from collections import OrderedDict

import mlflow
from mlflow import log_metric, log_param

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale, Resize

import flwr as fl
from pneumonia_network import PneumoniaNetwork


class PTLearner:
    def __init__(
        self, dataset_dir, lr=0.01, epochs=5, experiment_name="default-experiment"
    ):
        """Simple PyTorch Learner.
        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            epochs (int, optional): Epochs. Defaults to 5.
            dataset_dir (str): Path to the train/test dataset
            experiment_name (str, optional): Experiment name. Default is "default-experiment".

        Attributes:
            model_: PneumoniaNetwork model
            loss_: CrossEntropy loss
            optimizer_: Stochastic gradient descent
            train_dataset_: Training Dataset obj
            train_loader_: Training DataLoader
            test_dataset_: Testing Dataset obj
            test_loader_: Testing DataLoader
        """
        self._lr = lr
        self._epochs = epochs
        self._dataset_dir = dataset_dir
        self._experiment_name = experiment_name

        # Training setup
        self.model_ = PneumoniaNetwork()
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_.to(self.device_)
        self.loss_ = nn.CrossEntropyLoss()
        self.optimizer_ = SGD(self.model_.parameters(), lr=self._lr, momentum=0.9)

        # Data setup
        IMG_HEIGHT, IMG_WIDTH = 224, 224
        IMG_MEAN = 0.4818
        IMG_STD = 0.2357
        transforms = Compose(
            [
                Grayscale(),
                Resize((IMG_HEIGHT, IMG_WIDTH)),
                ToTensor(),
                Normalize(mean=(IMG_MEAN,), std=(IMG_STD,)),
            ]
        )
        self.train_dataset_, self.test_dataset_ = self.load_dataset(
            dataset_dir, transforms
        )
        self.train_loader_ = DataLoader(
            dataset=self.train_dataset_, batch_size=32, shuffle=True, drop_last=True
        )
        self.n_iterations = len(self.train_loader_)
        self.test_loader_ = DataLoader(
            dataset=self.test_dataset_, batch_size=100, shuffle=False
        )

    def load_dataset(self, data_dir, transforms):
        """Load dataset from {data_dir} directory. It is assumed that it contains two subdirectories 'train' and 'test'.

        Args:
            data_dir(str, optional): Data directory path
        """
        logger.info(f"Data dir: {data_dir}.")

        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "train"), transform=transforms
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "test"), transform=transforms
        )

        return train_dataset, test_dataset

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
                key=f"iteration_{self._iteration_num}/{self._experiment_name}/{key}",
                value=value,
            )

    def local_train(self, checkpoint):
        """Perform local training for a given number of epochs

        Args:
            checkpoint: Previous model checkpoint from where training has to be started.
        """
        if checkpoint:
            self.model_.load_state_dict(torch.load(checkpoint + "/model.pt"))

        mlflow_run = mlflow.active_run()

        # get Mlflow client and root run id
        mlflow_client = mlflow.tracking.client.MlflowClient()
        logger.debug(f"Root runId: {mlflow_run.data.tags.get('mlflow.rootRunId')}")
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")

        self.model_.train()
        logger.debug("Local training started")

        training_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0

        # Basic training
        for epoch in range(self._epochs):
            running_loss = 0.0
            num_of_batches_before_logging = 100

            for i, batch in enumerate(self.train_loader_):
                images, labels = batch[0].to(self.device_), batch[1].to(self.device_)
                self.optimizer_.zero_grad()

                predictions = self.model_(images)
                cost = self.loss_(predictions, labels)
                cost.backward()
                self.optimizer_.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i != 0 and i % num_of_batches_before_logging == 0:
                    training_loss = running_loss / num_of_batches_before_logging
                    logger.info(
                        f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, Training Loss: {training_loss}"
                    )

                    # log train loss
                    self.log_metrics(
                        mlflow_client,
                        root_run_id,
                        "Train Loss",
                        training_loss,
                        pipeline_level=True,
                    )

                    running_loss = 0.0

    def test(self):
        """Test the trained model and report test loss and accuracy"""
        self.model_.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader_:
                data, target = data.to(self.device_), target.to(self.device_)
                output = self.model_(data)
                test_loss += self.loss_(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader_.dataset)
        acc = correct / len(self.test_loader_.dataset)

        return test_loss, acc


# Define Flower client


class CustomFlowerClient(fl.client.NumPyClient):
    def __init__(self, pt_trainer):
        super().__init__()
        self.pt_trainer = pt_trainer

    def get_parameters(self, config):
        return [
            val.cpu().numpy() for _, val in self.pt_trainer.model_.state_dict().items()
        ]

    def set_parameters(self, parameters):
        params_dict = zip(self.pt_trainer.model_.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.pt_trainer.model_.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.pt_trainer.local_train(checkpoint=None)
        return self.get_parameters(config={}), len(self.pt_trainer.train_loader_), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.pt_trainer.test()
        return loss, len(self.pt_trainer.test_loader_), {"accuracy": accuracy}


def fetch_server_ip(
    mlflow_run, federation_identifier, target_run_tag="mlflow.rootRunId", timeout=600
):
    logger = logging.getLogger(__name__)

    mlflow_client = mlflow.tracking.client.MlflowClient()
    logger.info(f"run tags: {mlflow_run.data.tags}")
    logger.info(
        f"target tag {target_run_tag}={mlflow_run.data.tags.get(target_run_tag)}"
    )
    target_run_id = mlflow_run.data.tags.get(target_run_tag)

    server_ip = None
    fetch_start_time = time.time()
    federation_identifier_tag = f"{federation_identifier}.server"

    while server_ip is None:
        logger.info(f"Checking out tag server_ip...")
        mlflow_root_run = mlflow_client.get_run(target_run_id)

        if federation_identifier_tag in mlflow_root_run.data.tags:
            server_ip = mlflow_root_run.data.tags[federation_identifier_tag]
            logger.info(f"server_ip found: {server_ip}")

        if server_ip is None and (time.time() - fetch_start_time > timeout):
            raise RuntimeError("Could not fetch the tag within timeout.")
        else:
            time.sleep(10)

    return server_ip


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
        "--federation_identifier",
        type=str,
        required=True,
        help="a unique identifier for the group of clients and server to find each other",
    )
    parser.add_argument(
        "--client_data",
        type=str,
        required=True,
        help="Path to the pneumonia train/test data",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="The previous model checkpoint."
    )
    parser.add_argument(
        "--lr", type=float, required=False, help="Training algorithm's learning rate."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="Total number of epochs for local training.",
    )
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix."
    )

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

    # get Mlflow client and root run id
    mlflow_run = mlflow.start_run()
    server_ip = fetch_server_ip(mlflow_run, args.federation_identifier)

    # original class from pneumonia example
    trainer = PTLearner(
        dataset_dir=args.client_data,
        lr=args.lr,
        epochs=args.epochs,
        experiment_name=args.metrics_prefix,
    )

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=f"{server_ip}:8080", client=CustomFlowerClient(trainer)
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
