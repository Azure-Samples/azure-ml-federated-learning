"""Script for mock components."""
import argparse
import logging
import sys
import os
import io

import socket
import mlflow
import torch
import pandas as pd
from torch import nn
import torch.distributed as dist
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import models, transforms
from mlflow import log_metric, log_param
from PIL import Image


class BottomModel(nn.Module):
    def __init__(self) -> None:
        super(BottomModel, self).__init__()
        self._model = models.resnet18(pretrained=True)
        self._model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self._model = torch.nn.Sequential(*list(self._model.children())[:-1])

    def forward(self, x) -> torch.tensor:
        return self._model(x)


class TopModel(nn.Module):
    def __init__(self) -> None:
        super(TopModel, self).__init__()
        self._model = nn.Linear(512, 10)

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
        self.images = [img for img in os.listdir(self.root_dir) if img.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class TopDataset(Dataset):
    """Image dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(TopDataset, self).__init__()
        self.labels = pd.read_csv(csv_file, index_col=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.labels["label"][idx]


class MnistTrainer:
    def __init__(
        self,
        global_size,
        global_rank,
        global_group,
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
        self._global_group = global_group

        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Build model
        if self._global_rank == 0:
            self.model_ = TopModel()
            self.loss_ = nn.CrossEntropyLoss()
        else:
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
        if self._global_rank == 0:
            train_dataset = TopDataset(train_data_dir + "/train.csv")
            test_dataset = TopDataset(test_data_dir + "/test.csv")
        else:
            transformer = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )

            train_dataset = BottomDataset(train_data_dir, transformer)
            test_dataset = BottomDataset(train_data_dir, transformer)

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
        if self._global_rank == 0:
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

            for epoch in range(self._epochs):

                running_loss = 0.0
                running_acc = 0.0
                num_of_batches_before_logging = 100

                for i, data in enumerate(self.train_loader_):
                    data = data.to(self.device_)
                    self.optimizer_.zero_grad()

                    if self._global_rank != 0:
                        output = self.model_(data)
                        dist.send(output, 0, self._global_group)
                        gradient = torch.zeros_like(output).to(self.device_)
                        dist.recv(gradient, 0, self._global_group)
                        output.backward(gradient)
                        self.optimizer_.step()
                        continue

                    outputs = []
                    for j in range(1, self._global_size):
                        output = torch.zeros((data.shape[0], 512))
                        dist.recv(output, j, self._global_group)
                        outputs.append(output)

                    # Average all intermediate results
                    outputs = torch.stack(outputs).to(torch.float32)
                    outputs.requires_grad = True
                    outputs_avg = outputs.mean(dim=0)

                    predictions = self.model_(outputs_avg)
                    loss = self.loss_(predictions, data)
                    loss.backward(retain_graph=True)
                    gradients = torch.autograd.grad(loss, outputs)[0]
                    self.optimizer_.step()

                    for j in range(1, self._global_size):
                        dist.send(gradients[j - 1], j, self._global_group)

                    running_loss += loss.item() / data.shape[0]
                    running_acc += (
                        (predictions.argmax(dim=1) == data).to(float).mean().item()
                    )
                    if i != 0 and i % num_of_batches_before_logging == 0:
                        training_loss = running_loss / num_of_batches_before_logging
                        training_accuracy = running_acc / num_of_batches_before_logging
                        logger.info(
                            f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, Training Loss: {training_loss}, Accuracy: {training_accuracy}"
                        )

                        # log train loss
                        self.log_metrics(
                            mlflow_client,
                            root_run_id,
                            "Train Loss",
                            training_loss,
                        )

                        running_loss = 0.0
                        running_acc = 0.0

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
            for data in self.test_loader_:
                data = data.to(self.device_)

                if self._global_rank != 0:
                    output = self.model_(data)
                    dist.send(output, 0, self._global_group)
                    continue

                outputs = []
                for i in range(1, self._global_size):
                    output = torch.zeros((data.shape[0], 512))
                    dist.recv(output, i, self._global_group)
                    outputs.append(output)

                # Average all intermediate results
                outputs = torch.autograd.Variable(torch.stack(outputs))
                outputs = outputs.to(torch.float32).mean(dim=0)

                predictions = self.model_(outputs)
                test_loss += self.loss_(predictions, data).item()

                predictions = predictions.argmax(dim=1, keepdim=True)
                correct += predictions.eq(data.view_as(predictions)).sum().item()

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


def run(global_group, args):
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
        global_group=global_group,
    )
    trainer.execute(args.checkpoint)


def get_open_port():
    from contextlib import closing
    import socket

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def retrieve_host_address(mlflow_client, root_run_id, global_rank):
    host_ip, host_port = None, None
    if global_rank == 0:
        import socket

        host_port = get_open_port()
        local_hostname = socket.gethostname()
        host_ip = str(socket.gethostbyname(local_hostname))
    else:
        import time

        host_ip, host_port = None, None
        fetch_start_time = time.time()

        while host_ip is None or host_port is None:
            logger.info(f"Checking out tag aml_host_ip and aml_host_port...")
            mlflow_root_run = mlflow_client.get_run(root_run_id)

            if "aml_host_ip" in mlflow_root_run.data.tags:
                host_ip = mlflow_root_run.data.tags["aml_host_ip"]
                logger.info(f"host_ip found: {host_ip}")

            if "aml_host_port" in mlflow_root_run.data.tags:
                host_port = mlflow_root_run.data.tags["aml_host_port"]
                logger.info(f"host_port found: {host_port}")

            if (host_ip is None) and (time.time() - fetch_start_time > 600):
                raise RuntimeError("Could not fetch the tag within timeout.")
            else:
                time.sleep(1)
    return host_ip, host_port


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

    # communicate to clients through mlflow root (magic)
    with mlflow.start_run() as mlflow_run:
        mlflow_client = mlflow.tracking.client.MlflowClient()
        logger.info(f"run tags: {mlflow_run.data.tags}")
        logger.info(f"parent runId: {mlflow_run.data.tags.get('mlflow.parentRunId')}")
        root_run_id = mlflow_run.data.tags.get("mlflow.parentRunId")
        host_ip, host_port = retrieve_host_address(
            mlflow_client, root_run_id, args.global_rank
        )
        distributed_method = f"tcp://{host_ip}:{host_port}"

        if args.global_rank == 0:
            mlflow_client.set_tag(run_id=root_run_id, key="aml_host_ip", value=host_ip)
            mlflow_client.set_tag(
                run_id=root_run_id, key="aml_host_port", value=host_port
            )

        import socket

        local_hostname = socket.gethostname()
        local_ip = str(socket.gethostbyname(local_hostname))
        world_size, rank = int(args.global_size), int(args.global_rank)
        logger.info(f"Local IP: {local_ip}")
        logger.info(f"World size: {args.global_size}, Rank: {args.global_rank}")
        logger.info(distributed_method)

        # initialize the process group
        retries = 0
        while retries < 10:
            try:
                import datetime

                logger.info("Initializing process group...")
                global_group = dist.init_process_group(
                    dist.Backend.GLOO,
                    rank=rank,
                    world_size=world_size,
                    init_method=distributed_method,
                    timeout=datetime.timedelta(seconds=10000),
                )
                logger.info("Process group initialized")
                break
            except Exception as e:
                import time

                logger.exception(e)

                time.sleep(10)
                retries += 1

    print(f"Running script with arguments: {args}")
    run(global_group, args)

    # destroy the process group
    dist.destroy_process_group(global_group)


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
