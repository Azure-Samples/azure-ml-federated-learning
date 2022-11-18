# This file defining the learner was adapted from https://github.com/Azure/medical-imaging/blob/main/federated-learning/pneumonia-federated/custom/pt_learner.py to run directly on Azure ML without using NVFlare.
"""Script for training component."""
import argparse
import logging
import sys
import os.path

import mlflow
from mlflow import log_metric, log_param

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale, Resize

from pneumonia_network import PneumoniaNetwork


class PTLearner:
    def __init__(
        self,
        lr=0.01,
        epochs=5,
        dataset_dir: str = "pneumonia-alldata",
        experiment_name="default-experiment",
        iteration_num=1,
        model_path=None,
    ):
        """Simple PyTorch Learner.
        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            epochs (int, optional): Epochs. Defaults to 5.
            dataset_dir (str, optional): Name of data asset in Azure ML. Defaults to "pneumonia-alldata".
            experiment_name (str, optional): Experiment name. Default is "default-experiment".
            iteration_num (int, optional): Iteration number. Defaults to 1
            model_path (str, optional): where in the output directory to save the model. Defaults to None.

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
        self._iteration_num = iteration_num
        self._model_path = model_path

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

    def log_params(self, client, run_id):
        client.log_param(
            run_id=run_id, key=f"learning_rate {self._experiment_name}", value=self._lr
        )
        client.log_param(
            run_id=run_id, key=f"epochs {self._experiment_name}", value=self._epochs
        )
        client.log_param(
            run_id=run_id,
            key=f"loss {self._experiment_name}",
            value=self.loss_.__class__.__name__,
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

        with mlflow.start_run() as mlflow_run:

            # get Mlflow client and root run id
            mlflow_client = mlflow.tracking.client.MlflowClient()
            logger.debug(f"Root runId: {mlflow_run.data.tags.get('mlflow.rootRunId')}")
            root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")

            # log params
            self.log_params(mlflow_client, root_run_id)

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

                    images, labels = batch[0].to(self.device_), batch[1].to(
                        self.device_
                    )
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
                        )

                        running_loss = 0.0

                # compute test metrics
                test_loss, test_acc = self.test()

                # log test metrics after each epoch
                self.log_metrics(mlflow_client, root_run_id, "Test Loss", test_loss)
                self.log_metrics(mlflow_client, root_run_id, "Test Accuracy", test_acc)

                logger.info(
                    f"Epoch: {epoch}, Test Loss: {test_loss} and Test Accuracy: {test_acc}"
                )

            # log metrics at the pipeline level
            self.log_metrics(
                mlflow_client,
                root_run_id,
                "Train Loss",
                training_loss,
                pipeline_level=True,
            )
            self.log_metrics(
                mlflow_client, root_run_id, "Test Loss", test_loss, pipeline_level=True
            )
            self.log_metrics(
                mlflow_client,
                root_run_id,
                "Test Accuracy",
                test_acc,
                pipeline_level=True,
            )

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

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of data asset in Azure ML.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="The previous model checkpoint."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Where to write the model output."
    )
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix."
    )
    parser.add_argument(
        "--iteration_num", type=int, required=False, help="Iteration number."
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

    return parser


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """
    trainer = PTLearner(
        dataset_dir=args.dataset_name,
        lr=args.lr,
        epochs=args.epochs,
        experiment_name=args.metrics_prefix,
        iteration_num=args.iteration_num,
        model_path=args.model + "/model.pt",
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
