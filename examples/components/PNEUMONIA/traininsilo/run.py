# This file defining the learner was adapted from https://github.com/Azure/medical-imaging/blob/main/federated-learning/pneumonia-federated/custom/pt_learner.py to run directly on Azure ML without using NVFlare.
"""Script for training component."""
import argparse
import logging
import sys
import os.path
from distutils.util import strtobool

import mlflow
from mlflow import log_metric, log_param

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale, Resize

from pneumonia_network import PneumoniaNetwork

# DP
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class PTLearner:
    def __init__(
        self,
        lr=0.01,
        epochs=5,
        dp=False,
        dp_target_epsilon=50.0,
        dp_target_delta=1e-5,
        dp_max_grad_norm=1.0,
        total_num_of_iterations=1,
        dataset_dir: str = "pneumonia-alldata",
        experiment_name="default-experiment",
        iteration_num=1,
        model_path=None,
        device_id=None,
        distributed=False,
    ):
        """Simple PyTorch Learner.
        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            epochs (int, optional): Epochs. Defaults to 5.
            dp (bool, optional): Differential Privacy. Default is False (Note: dp, dp_target_epsilon, dp_target_delta, dp_max_grad_norm, and total_num_of_iterations are defined for the only purpose of DP and can be ignored when users don't want to use Differential Privacy)
            dp_target_epsilon (float, optional): DP target epsilon. Default is 50.0
            dp_target_delta (float, optional): DP target delta. Default is 1e-5
            dp_max_grad_norm (float, optional): DP max gradient norm. Default is 1.0
            total_num_of_iterations (int, optional): Total number of iterations. Defaults to 1
            dataset_dir (str, optional): Name of data asset in Azure ML. Defaults to "pneumonia-alldata".
            experiment_name (str, optional): Experiment name. Default is "default-experiment".
            iteration_num (int, optional): Iteration number. Defaults to 1.
            model_path (str, optional): where in the output directory to save the model. Defaults to None.
            device_id (int, optional): Device id to run training on. Default to None.
            distributed (bool, optional): Whether to run distributed training. Default to False.

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

        # Training setup
        self.model_ = PneumoniaNetwork()
        self.model_.to(self.device_)
        if self._distributed:
            self.model_ = DDP(
                self.model_,
                device_ids=[self._rank] if self._rank is not None else None,
                output_device=self._rank,
            )
        self.loss_ = nn.CrossEntropyLoss()

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

        if self._distributed:
            logger.info("Setting up distributed samplers.")
            self.train_sampler_ = DistributedSampler(self.train_dataset_)
            self.test_sampler_ = DistributedSampler(self.test_dataset_)
        else:
            self.train_sampler_ = None
            self.test_sampler_ = None

        self.train_loader_ = DataLoader(
            dataset=self.train_dataset_,
            batch_size=32,
            shuffle=(not self._distributed),
            drop_last=True,
            sampler=self.train_sampler_,
        )
        self.n_iterations = len(self.train_loader_)
        self.test_loader_ = DataLoader(
            dataset=self.test_dataset_,
            batch_size=100,
            shuffle=False,
            sampler=self.test_sampler_,
        )

        logger.info(f"Train loader steps: {len(self.train_loader_)}")
        logger.info(f"Test loader steps: {len(self.test_loader_)}")

        # DP
        logger.info(f"DP: {dp}")
        if dp:
            if not ModuleValidator.is_valid(self.model_):
                self.model_ = ModuleValidator.fix(self.model_)

        self.optimizer_ = SGD(self.model_.parameters(), lr=self._lr, momentum=0.9)

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
                key=f"loss {self._experiment_name}",
                value=self.loss_.__class__.__name__,
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
                    key=f"iteration_{self._iteration_num}/{self._experiment_name}/{key}",
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
            # get Mlflow client and root run id
            mlflow_client = mlflow.tracking.client.MlflowClient()
            root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
            logger.debug(f"Root runId: {root_run_id}")

            # log params
            self.log_params(mlflow_client, root_run_id)

            logger.debug("Local training started")

            training_loss = 0.0
            test_loss = 0.0
            test_acc = 0.0

            # Basic training
            for epoch in range(self._epochs):
                running_loss = 0.0
                num_of_batches_before_logging = 100
                self.model_.train()

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
                        if not self._distributed or self._rank == 0:
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
                if not self._distributed or self._rank == 0:
                    self.log_metrics(mlflow_client, root_run_id, "Test Loss", test_loss)
                    self.log_metrics(
                        mlflow_client, root_run_id, "Test Accuracy", test_acc
                    )

                logger.info(
                    f"Epoch: {epoch}, Test Loss: {test_loss} and Test Accuracy: {test_acc}"
                )

            # log metrics at the pipeline level
            if not self._distributed or self._rank == 0:
                self.log_metrics(
                    mlflow_client,
                    root_run_id,
                    "Train Loss",
                    training_loss,
                    pipeline_level=True,
                )
                self.log_metrics(
                    mlflow_client,
                    root_run_id,
                    "Test Loss",
                    test_loss,
                    pipeline_level=True,
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
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
    elif int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("gloo")

    trainer = PTLearner(
        dataset_dir=args.dataset_name,
        lr=args.lr,
        epochs=args.epochs,
        dp=args.dp,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_target_delta=args.dp_target_delta,
        dp_max_grad_norm=args.dp_max_grad_norm,
        total_num_of_iterations=args.total_num_of_iterations,
        experiment_name=args.metrics_prefix,
        iteration_num=args.iteration_num,
        model_path=args.model + "/model.pt",
        device_id=int(os.environ["RANK"]),
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
