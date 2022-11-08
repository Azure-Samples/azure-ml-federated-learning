"""Script for mock components."""
import argparse
import logging
import sys
import os.path

import mlflow
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import models, datasets, transforms
from mlflow import log_metric, log_param

# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale, Resize
# from torch.utils.tensorboard import SummaryWriter

# from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
# from nvflare.apis.fl_constant import FLContextKey, ReturnCode, ReservedKey
# from nvflare.apis.fl_context import FLContext
# from nvflare.apis.shareable import Shareable, make_reply
# from nvflare.apis.signal import Signal
# from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, \
#                                         make_model_learnable, model_learnable_to_dxo
# from nvflare.app_common.abstract.learner_spec import Learner
# from nvflare.app_common.app_constant import AppConstants
# from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants
from pneumonia_network import PneumoniaNetwork
# from azureml.core import Workspace, Dataset


class PTLearner:

    def __init__(
        self,
        lr=0.01,
        epochs=5,
        exclude_vars=None,
        dataset_dir: str = "pneumonia-reduced",
        analytic_sender_id="analytic_sender",
        experiment_name="default-experiment",
        iteration_num=1,
        model_path=None,
    ):
        """Simple PyTorch Learner.
        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            exclude_vars (list): List of variables to exclude during model loading.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
            If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            experiment_name (str, optional): Experiment name. Default is default-experiment
            iteration_num (int, optional): Iteration number. Defaults to 1
            model_path (str, optional): where in the output directory to save the model

        Attributes:
            model_: PneumoniaNetwork model
            loss_: CrossEntropy loss
            optimizer_: Stochastic gradient descent
            train_dataset_: Training Dataset obj
            train_loader_: Training DataLoader
            test_dataset_: Testing Dataset obj
            test_loader_: Testing DataLoader
        """
        self.lr = lr
        self.epochs = epochs
        self.exclude_vars = exclude_vars
        # self.dataset_name = dataset_name
        self.analytic_sender_id = analytic_sender_id
        self._experiment_name = experiment_name
        self._iteration_num = iteration_num

    # def initialize(self, parts: dict, fl_ctx: FLContext):
        # Training setup
        self.model_ = PneumoniaNetwork()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_.to(self.device)
        self._model_path = model_path
        self.loss_ = nn.CrossEntropyLoss()
        self.optimizer_ = SGD(self.model_.parameters(), lr=self.lr, momentum=0.9)

        # self.workspace = Workspace.from_config()
        # print(self.workspace)
        print('model', self.model_)

        IMG_HEIGHT, IMG_WIDTH = 224, 224
        IMG_MEAN = 0.4818
        IMG_STD = 0.2357

        transforms = Compose([Grayscale(),
                              Resize((IMG_HEIGHT, IMG_WIDTH)),
                              ToTensor(),
                              Normalize(mean=(IMG_MEAN,), std=(IMG_STD,))
                              ])

        # pneumonia_dataset = Dataset.get_by_name(
        #     self.workspace, self.dataset_name)
        # pneumonia_dataset.download(target_path=os.path.join(
        #     os.path.expanduser('~'), 'data'), overwrite=True)
        # self.train_dataset = ImageFolder(root=os.path.join(os.path.expanduser(
        #     '~'), 'data', 'pneumonia-2class-reduced', 'train'), transform=transforms)

        self.train_dataset_, self.test_dataset_ = self.load_dataset(dataset_dir)

        self.train_loader_ = DataLoader(
            dataset=self.train_dataset_, batch_size=32, shuffle=True, drop_last=True)

        self.n_iterations = len(self.train_loader)

        # self.test_dataset = ImageFolder(root=os.path.join(os.path.expanduser(
        #     '~'), 'data', 'pneumonia-2class-reduced', 'test'), transform=transforms)
        self.test_loader_ = DataLoader(
            dataset=self.test_dataset_, batch_size=100, shuffle=False)

        # # Setup the persistence manager to save PT model.
        # # The default training configuration is used by persistence manager in case no initial model is found.
        # self.default_train_conf = {
        #     "train": {"model": type(self.model_).__name__}}
        # print('default_train_conf', self.default_train_conf)
        # self.persistence_manager = PTModelPersistenceFormatManager(
        #     data=self.model_.state_dict(), default_train_conf=self.default_train_conf)

        # # Tensorboard streaming setup
        # # user configuration from config_fed_client.json
        # self.writer = parts.get(self.analytic_sender_id)
        # if not self.writer:  # else use local TensorBoard writer only
        #     self.writer = SummaryWriter(fl_ctx.get_prop(FLContextKey.APP_ROOT))

    def load_dataset(self, data_dir):
        """Load dataset from {data_dir}

        Args:
            data_dir(str, optional): Data directory path
        """
        logger.info(f"Data dir: {data_dir}.")
        transformer = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transformer)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transformer)

        return train_dataset, test_dataset

    # def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
    #     # Get model weights
    #     try:
    #         dxo = from_shareable(data)
    #     except:
    #         self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
    #         return make_reply(ReturnCode.BAD_TASK_DATA)

    #     # Ensure data kind is weights.
    #     if not dxo.data_kind == DataKind.WEIGHTS:
    #         self.log_error(
    #             fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
    #         return make_reply(ReturnCode.BAD_TASK_DATA)

    #     # Convert weights to tensor. Run training
    #     torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
    #     self.local_train(fl_ctx, torch_weights, abort_signal)

    #     # Check the abort_signal after training.
    #     # local_train returns early if abort_signal is triggered.
    #     if abort_signal.triggered:
    #         return make_reply(ReturnCode.TASK_ABORTED)

    #     # Save the local model after training.
    #     # self.save_local_model(fl_ctx)

    #     # Get the new state dict and send as weights
    #     new_weights = self.model_.state_dict()
    #     new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

    #     outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
    #                        meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.n_iterations})
    #     return outgoing_dxo.to_shareable()

    def log_params(self, client, run_id):
        client.log_param(
            run_id=run_id, key=f"learning_rate {self._experiment_name}", value=self._lr
        )
        client.log_param(
            run_id=run_id, key=f"epochs {self._experiment_name}", value=self._epochs
        )
        # client.log_param(
        #     run_id=run_id,
        #     key=f"batch_size {self._experiment_name}",
        #     value=self._batch_size,
        # )
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
            for epoch in range(self.epochs):
                # self.model_.train()
                running_loss = 0.0
                num_of_batches_before_logging = 100

                for i, batch in enumerate(self.train_loader_):

                    images, labels = batch[0].to(
                        self.device), batch[1].to(self.device)
                    self.optimizer_.zero_grad()

                    predictions = self.model_(images)
                    cost = self.loss_(predictions, labels)
                    cost.backward()
                    self.optimizer_.step()

                    running_loss += (cost.cpu().detach().numpy()/images.size()[0])
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

                    # # Stream training loss at each step
                    # current_step = len(self.train_loader) * epoch + i
                    # self.writer.add_scalar("train_loss", cost.item(), current_step)

                # # Stream validation accuracy at the end of each epoch
                # metric = self.local_validate(self.test_loader, abort_signal)
                # self.writer.add_scalar("validation_accuracy", metric, epoch)
                
                # test_loss, test_acc = self.test()
                test_loss, test_acc = 0, 0


                # log test metrics after each epoch
                self.log_metrics(mlflow_client, root_run_id, "Test Loss", test_loss)
                self.log_metrics(mlflow_client, root_run_id, "Test Accuracy", test_acc)

                logger.info(
                    f"Epoch: {epoch}, Test Loss: {test_loss} and Test Accuracy: {test_acc}"
                )

    # def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
    #     run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
    #         fl_ctx.get_prop(ReservedKey.RUN_NUM))
    #     models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
    #     if not os.path.exists(models_dir):
    #         return None
    #     model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

    #     self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
    #                                                                default_train_conf=self.default_train_conf)
    #     ml = self.persistence_manager.to_model_learnable(
    #         exclude_vars=self.exclude_vars)

    #     # Get the model parameters and create dxo from it
    #     dxo = model_learnable_to_dxo(ml)
    #     return dxo.to_shareable()

    # def validate(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
    #     model_owner = "?"
    #     try:
    #         try:
    #             dxo = from_shareable(data)
    #         except:
    #             self.log_error(
    #                 fl_ctx, "Error in extracting dxo from shareable.")
    #             return make_reply(ReturnCode.BAD_TASK_DATA)

    #         # Ensure data_kind is weights.
    #         if not dxo.data_kind == DataKind.WEIGHTS:
    #             self.log_exception(
    #                 fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
    #             return make_reply(ReturnCode.BAD_TASK_DATA)

    #         if isinstance(dxo.data, ModelLearnable):
    #             dxo.data = dxo.data[ModelLearnableKey.WEIGHTS]

    #         # Extract weights and ensure they are tensor.
    #         model_owner = data.get_header(AppConstants.MODEL_OWNER, "?")
    #         weights = {k: torch.as_tensor(v, device=self.device)
    #                    for k, v in dxo.data.items()}

    #         self.model_.load_state_dict(weights)

    #         # Get validation accuracy
    #         val_accuracy = self.local_validate(weights, abort_signal)
    #         if abort_signal.triggered:
    #             return make_reply(ReturnCode.TASK_ABORTED)

    #         self.log_info(fl_ctx, f"Accuracy when validating {model_owner}'s model on"
    #                       f" {fl_ctx.get_identity_name()}"f's data: {val_accuracy}')

    #         dxo = DXO(data_kind=DataKind.METRICS,
    #                   data={'val_acc': val_accuracy})
    #         return dxo.to_shareable()
    #     except:
    #         self.log_exception(
    #             fl_ctx, f"Exception in validating model from {model_owner}")
    #         return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    # def local_validate(self, weights, abort_signal):
    #     self.model_.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for i, (images, labels) in enumerate(self.test_loader):
    #             if abort_signal.triggered:
    #                 return 0

    #             images, labels = images.to(self.device), labels.to(self.device)
    #             output = self.model_(images)

    #             _, pred_label = torch.max(output, 1)

    #             correct += (pred_label == labels).sum().item()
    #             total += images.size()[0]
    #         metric = correct/float(total)
    #     return metric

    # def save_local_model(self, fl_ctx: FLContext):
    #     run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
    #         fl_ctx.get_prop(ReservedKey.RUN_NUM))
    #     models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
    #     if not os.path.exists(models_dir):
    #         os.makedirs(models_dir)
    #     model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

    #     ml = make_model_learnable(self.model_.state_dict(), {})
    #     self.persistence_manager.update(ml)
    #     torch.save(self.persistence_manager.to_persistence_dict(), model_path)

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

    parser.add_argument("--dataset_name", type=str, required=True, help="")
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
    parser.add_argument("--exclude_vars", type=str, required=False, help="list of variables to exclude during model loading")
    parser.add_argument("--analytic_sender_id", type=str, required=False, help="id of `AnalyticsSender` if configured as a client component")
    
    return parser


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    # logger.info("Instantiate trainer...")
    trainer = PTLearner(
        dataset_dir=args.dataset_name,        
        lr=args.lr,
        epochs=args.epochs,
        experiment_name=args.metrics_prefix,
        iteration_num=args.iteration_num,
        model_path=args.model + "/model.pt",
    )
    # logger.info(trainer)
    # logger.info("Trainer has been instantiated.")
    # logger.info("Execute trainer...")
    trainer.execute(args.checkpoint)
    # logger.info("Trainer has been executed.")


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
    #logger.setLevel(logging.DEBUG) # Restore to debug level
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    #handler.setLevel(logging.DEBUG) # Restore to debug level
    handler.setLevel(logging.INFO)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    main()
