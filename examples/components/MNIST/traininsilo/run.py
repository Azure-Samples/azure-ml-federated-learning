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
from distutils.util import strtobool

# DP
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class MnistTrainer:
    def __init__(
        self,
        train_data_dir="./",
        test_data_dir="./",
        model_path=None,
        lr=0.01,
        epochs=1,
        batch_size=64,
        dp=False,
        dp_target_epsilon=50.0,
        dp_target_delta=1e-5,
        dp_max_grad_norm=1.0,
        total_num_of_iterations=1,
        experiment_name="default-experiment",
        iteration_num=1,
    ):
        """MNIST Trainer trains RESNET18 model on the MNIST dataset.

        Args:
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 1
            batch_size (int, optional): DataLoader batch size. Defaults to 64
            dp (bool, optional): Differential Privacy. Default is False (Note: dp, dp_target_epsilon, dp_target_delta, dp_max_grad_norm, and total_num_of_iterations are defined for the only purpose of DP and can be ignored when users don't want to use Differential Privacy)
            dp_target_epsilon (float, optional): DP target epsilon. Default is 50.0
            dp_target_delta (float, optional): DP target delta. Default is 1e-5
            dp_max_grad_norm (float, optional): DP max gradient norm. Default is 1.0
            total_num_of_iterations (int, optional): Total number of iterations. Defaults to 1
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

        # Data Loader
        self.train_dataset_, self.test_dataset_ = self.load_dataset(
            train_data_dir, test_data_dir
        )
        self.train_loader_ = DataLoader(
            self.train_dataset_, batch_size=batch_size, shuffle=True
        )
        self.test_loader_ = DataLoader(
            self.test_dataset_, batch_size=batch_size, shuffle=True
        )

        # DP
        logger.info(f"DP: {dp}")
        if dp:
            if not ModuleValidator.is_valid(self.model_):
                self.model_ = ModuleValidator.fix(self.model_)

        self.optimizer_ = SGD(self.model_.parameters(), lr=lr, momentum=0.9)

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
                        self.log_metrics(
                            mlflow_client,
                            root_run_id,
                            "Train Loss",
                            training_loss,
                        )

                        running_loss = 0.0

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

    parser.add_argument("--train_data", type=str, required=True, help="")
    parser.add_argument("--test_data", type=str, required=True, help="")
    parser.add_argument("--checkpoint", type=str, required=False, help="")
    parser.add_argument("--model", type=str, required=True, help="")
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

    trainer = MnistTrainer(
        train_data_dir=args.train_data,
        test_data_dir=args.test_data,
        model_path=args.model + "/model.pt",
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dp=args.dp,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_target_delta=args.dp_target_delta,
        dp_max_grad_norm=args.dp_max_grad_norm,
        total_num_of_iterations=args.total_num_of_iterations,
        experiment_name=args.metrics_prefix,
        iteration_num=args.iteration_num,
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
