"""Script for mock components."""
import argparse
import logging
import sys

import mlflow
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import models, datasets, transforms
from mlflow import log_metric, log_param


class MnistTrainer:
    def __init__(
        self,
        train_data_dir="./",
        test_data_dir="./",
        model_path=None,
        lr=0.01,
        epochs=1,
        batch_size=64,
    ):
        """MNIST Trainer trains RESNET18 model on the MNIST dataset.

        Args:
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 1
            batch_size (int, optional): DataLoader batch size. Defaults to 64.

        Attributes:
            model_: RESNET18 model
            loss_: CrossEntropy loss
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

        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model_ = models.resnet18(pretrained=True)
        self.model_.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        num_ftrs = self.model_.fc.in_features
        self.model_.fc = nn.Linear(num_ftrs, 10)
        self.model_.to(self.device_)
        self._model_path = model_path

        self.loss_ = nn.CrossEntropyLoss()
        self.optimizer_ = SGD(self.model_.parameters(), lr=lr, momentum=0.9)

        self.train_dataset_, self.test_dataset_ = self.load_dataset(
            train_data_dir, test_data_dir
        )
        self.train_loader_ = DataLoader(
            self.train_dataset_, batch_size=batch_size, shuffle=True
        )
        self.test_loader_ = DataLoader(
            self.test_dataset_, batch_size=batch_size, shuffle=True
        )

    def load_dataset(self, train_data_dir, test_data_dir):
        """Load dataset from {train_data_dir} and {test_data_dir}

        Args:
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            batch_size (int, optional): DataLoader batch size. Defaults to 64.
        """
        logger.info(f"Train data dir: {train_data_dir}, Test data dir: {test_data_dir}")
        transformer = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.ImageFolder(train_data_dir, transformer)
        test_dataset = datasets.ImageFolder(test_data_dir, transformer)

        return train_dataset, test_dataset

    def log_params(self):
        log_param("learning_rate", self._lr)
        log_param("epochs", self._epochs)
        log_param("batch_size", self._batch_size)
        log_param("loss", self.loss_.__class__.__name__)
        log_param("optimizer", self.optimizer_.__class__.__name__)

    def local_train(self, checkpoint):
        """Perform local training for a given number of epochs

        Args:
            checkpoint: Previous model checkpoint from where training has to be started.
        """

        if checkpoint:
            self.model_.load_state_dict(torch.load(checkpoint + "/model.pt"))

        with mlflow.start_run() as mlflow_run:
            self.model_.train()
            logger.debug("Local training started")

            self.log_params()

            for epoch in range(self._epochs):

                running_loss = 0.0
                num_of_iter_before_logging = 3000

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
                    if i != 0 and i % num_of_iter_before_logging == 0:
                        print(
                            f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, "
                            f"Loss: {running_loss/num_of_iter_before_logging}"
                        )
                        log_metric(
                            "Train Loss", f"{running_loss/num_of_iter_before_logging}"
                        )
                        running_loss = 0.0

                test_loss, test_acc = self.test()
                log_metric("Test Loss", f"{test_loss}", step=epoch)
                log_metric("Test Accuracy", f"{test_acc}", step=epoch)
                logger.info(
                    f"Epoch: {epoch}, Test Loss: {test_loss} and Test Accuracy: {test_acc}"
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

    parser.add_argument("--train_data", type=str, required=True, help="")
    parser.add_argument("--test_data", type=str, required=True, help="")
    parser.add_argument("--checkpoint", type=str, required=False, help="")
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument(
        "--lr", type=float, required=False, help="Training algorithm's learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="Total number of rounds for local training",
    )
    parser.add_argument("--batch_size", type=int, required=False, help="Batch Size")
    return parser


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    trainer = MnistTrainer(
        train_data_dir=args.train_data,
        test_data_dir=args.test_data,
        model_path=args.model + "/model.pt",
        lr=args.lr,
        epochs=args.epochs,
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
