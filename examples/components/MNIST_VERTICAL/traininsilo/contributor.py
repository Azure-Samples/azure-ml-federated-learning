"""Script for mock components."""
import argparse
import logging
import sys
import os
from aml_comm import AMLComm

import mlflow
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import models, transforms
from PIL import Image


class BottomModel(nn.Module):
    def __init__(self) -> None:
        super(BottomModel, self).__init__()
        self._model = models.resnet18(pretrained=True, progress=False)
        self._model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self._model = torch.nn.Sequential(*list(self._model.children())[:-1])

    def forward(self, x) -> torch.tensor:
        return self._model(x)


class BottomDataset(Dataset):
    """Image dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(BottomDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images = list(
            sorted(
                [
                    int(img.split(".")[0])
                    for img in os.listdir(self.root_dir)
                    if img.endswith(".jpg")
                ]
            )
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.images[idx]) + ".jpg")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class MnistTrainer:
    def __init__(
        self,
        global_size,
        global_rank,
        global_comm,
        train_data_dir="./",
        test_data_dir="./",
        model_path=None,
        lr=0.01,
        epochs=1,
        batch_size=64,
        experiment_name="default-experiment",
        iteration_num=1,
    ):
        """MNIST Trainer trains RESNET18 model on the MNIST dataset.

        Args:
            workers_num (int): Number of processes involved in VFL
            rank (int): Id of current process
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 1
            batch_size (int, optional): DataLoader batch size. Defaults to 64
            experiment_name (str, optional): Experiment name. Default is default-experiment
            iteration_num (int, optional): Iteration number. Defaults to 1

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
        self._experiment_name = experiment_name
        self._iteration_num = iteration_num
        self._global_size = global_size
        self._global_rank = global_rank
        self._global_comm = global_comm

        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model_ = BottomModel()

        self.train_dataset_, self.test_dataset_ = self.load_dataset(
            train_data_dir, test_data_dir
        )
        self.train_loader_ = DataLoader(
            self.train_dataset_, batch_size=batch_size, shuffle=False
        )
        self.test_loader_ = DataLoader(
            self.test_dataset_, batch_size=batch_size, shuffle=False
        )

        self.model_.to(self.device_)
        self._model_path = model_path

        self.optimizer_ = SGD(self.model_.parameters(), lr=self._lr, momentum=0.9)

    def load_dataset(self, train_data_dir, test_data_dir):
        """Load dataset from {train_data_dir} and {test_data_dir}

        Args:
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
        """
        logger.info(f"Train data dir: {train_data_dir}, Test data dir: {test_data_dir}")
        transformer = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        train_dataset = BottomDataset(train_data_dir, transformer)
        test_dataset = BottomDataset(test_data_dir, transformer)

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

            for epoch in range(self._epochs):

                running_loss = 0.0
                running_acc = 0.0
                num_of_batches_before_logging = 100

                for i, data in enumerate(self.train_loader_):
                    data = data.to(self.device_)
                    self.optimizer_.zero_grad()

                    output = self.model_(data)
                    self._global_comm.send(output, 0)
                    gradient = self._global_comm.recv(0).to(self.device_)
                    output.backward(gradient)
                    self.optimizer_.step()

                self.test()

    def test(self):
        """Test the trained model and report test loss and accuracy"""
        self.model_.eval()

        with torch.no_grad():
            for data in self.test_loader_:
                data = data.to(self.device_)

                output = self.model_(data)
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
    parser.add_argument("--model", type=str, required=True, help="")
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
        "--local_size",
        type=int,
        required=True,
        help="Number of silos",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        required=True,
        help="Index of the current silo",
    )
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    parser.add_argument(
        "--iteration_num", type=int, required=False, help="Iteration number"
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


def run(global_comm, args):
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
        experiment_name=args.metrics_prefix,
        iteration_num=args.iteration_num,
        global_size=args.global_size,
        global_rank=args.global_rank,
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

    logger.info(args.train_data)
    logger.info(args.test_data)

    root_run_id = None
    with mlflow.start_run() as mlflow_run:
        logger.info(f"run tags: {mlflow_run.data.tags}")
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        logger.info(f"root runId: {root_run_id}")

    global_comm = AMLComm(args.global_rank, args.global_size, root_run_id)

    print(f"Running script with arguments: {args}")
    run(global_comm, args)

    # destroy communication group
    global_comm.close()


if __name__ == "__main__":
    # Set logging to sys.out
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    # run training
    main()
