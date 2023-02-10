import argparse
import logging
import sys
import os

from transformers import (
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_scheduler,
    PreTrainedTokenizer,
)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk
from torch.optim import AdamW
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import evaluate
import mlflow
import torch
from distutils.util import strtobool

# DP
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import List, Dict, Union


# Custom data collator for differential privacy
class DataCollatorForPrivateTokenClassification(DataCollatorForTokenClassification):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        # Huggingface's default way of constructing position_ids is not compatible with Opacus
        # since Opacus is not able to deduce the batch size from the input. Here we manually
        # generate a position_ids tensor which has the same values as Huggingface's default tensor
        # but it is constructed in a way that is compatile with Opacus by using expand_as.
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


class NERTrainer:
    def __init__(
        self,
        tokenizer_name,
        model_name,
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
        device_id=None,
        distributed=False,
    ):
        """NER Trainer trains BERT-base model (default) on the MultiNERD dataset.

        Args:
            tokenizer_name(str): Tokenizer name.
            model_name(str): Model name.
            train_data_dir(str, optional): Training data directory path.
            test_data_dir(str, optional): Testing data directory path.
            model_path (str, optional): Model path. Defaults to None.
            lr (float, optional): Learning rate. Defaults to 0.01.
            epochs (int, optional): number of epochs. Defaults to 1.
            batch_size (int, optional): DataLoader batch size. Defaults to 64.
            dp (bool, optional): Differential Privacy. Default is False (Note: dp, dp_target_epsilon, dp_target_delta, dp_max_grad_norm, and total_num_of_iterations are defined for the only purpose of DP and can be ignored when users don't want to use Differential Privacy)
            dp_target_epsilon (float, optional): DP target epsilon. Default is 50.0
            dp_target_delta (float, optional): DP target delta. Default is 1e-5
            dp_max_grad_norm (float, optional): DP max gradient norm. Default is 1.0
            total_num_of_iterations (int, optional): Total number of iterations. Defaults to 1
            experiment_name (str, optional): Experiment name. Default is default-experiment.
            iteration_num (int, optional): Iteration number. Defaults to 1.
            device_id (int, optional): Device id to run training on. Default to None.
            distributed (bool, optional): Whether to run distributed training. Default to False.

        Attributes:
            model_: Huggingface bert-base (default) pretrained model
            optimizer_: Stochastic gradient descent
            train_loader_: Training DataLoader
            test_loader_: Testing DataLoader
            labelToId_: label to ID mapping
            idToLabel_: ID to label mapping
            metric_: Evaluation metrics
            device_: Either cpu or gpu
        """

        # training params
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._experiment_name = experiment_name
        self._iteration_num = iteration_num
        self._model_path = model_path
        self._distributed = distributed

        # device
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

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # dataset and data loader
        data_collator = DataCollatorForPrivateTokenClassification(tokenizer=tokenizer)
        train_dataset, test_dataset = self.load_dataset(train_data_dir, test_data_dir)

        if self._distributed:
            logger.info("Setting up distributed samplers.")
            self.train_sampler_ = DistributedSampler(train_dataset)
            self.test_sampler_ = DistributedSampler(test_dataset)
        else:
            self.train_sampler_ = None
            self.test_sampler_ = None

        self.train_loader_ = DataLoader(
            train_dataset,
            shuffle=(not self._distributed),
            collate_fn=data_collator,
            batch_size=self._batch_size,
            sampler=self.train_sampler_,
        )
        self.test_loader_ = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size=self._batch_size,
            sampler=self.test_sampler_,
        )

        logger.info(f"Train loader steps: {len(self.train_loader_)}")
        logger.info(f"Test loader steps: {len(self.test_loader_)}")

        # training params
        self.labelToId_ = pd.read_json("./labels.json", typ="series").to_dict()
        self.idToLabel_ = {val: key for key, val in self.labelToId_.items()}
        self.model_ = AutoModelForTokenClassification.from_pretrained(
            model_name,
            id2label=self.idToLabel_,
            label2id=self.labelToId_,
        )
        trainable_layers = [self.model_.bert.encoder.layer[-1], self.model_.classifier]
        trainable_params = 0
        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_params += p.numel()
        logger.info(f"Trainable parameters: {trainable_params}")

        self.model_.to(self.device_)
        if self._distributed:
            self.model_ = DDP(
                self.model_,
                device_ids=[self._rank] if self._rank is not None else None,
                output_device=self._rank,
            )
        self.metric_ = evaluate.load("seqeval")

        # DP
        logger.info(f"DP: {dp}")
        if dp:
            self.model_.train()
            if not ModuleValidator.is_valid(self.model_):
                self.model_ = ModuleValidator.fix(self.model_)

        self.optimizer_ = AdamW(self.model_.parameters(), lr=2e-5)

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

        Returns:
            Train dataset
            Test datset
        """
        logger.info(f"Train data dir: {train_data_dir}, Test data dir: {test_data_dir}")
        train_dataset = load_from_disk(train_data_dir)
        test_dataset = load_from_disk(test_data_dir)

        return train_dataset, test_dataset

    def log_params(self, client, run_id):
        """Log parameters to the mlflow metrics

        Args:
            client (MlflowClient): Mlflow Client where metrics will be logged
            run_id (str): Run ID

        Returns:
            None
        """
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
                key=f"optimizer {self._experiment_name}",
                value=self.optimizer_.__class__.__name__,
            )

    def log_metrics(self, client, run_id, key, value, pipeline_level=False):
        """Log mlflow metrics

        Args:
            client (MlflowClient): Mlflow Client where metrics will be logged
            run_id (str): Run ID
            key (str): metrics x-axis label
            value (float): metrics y-axis values
            pipeline_level (bool): whether the metrics is logged at the pipeline or the job level

        Returns:
            None
        """
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

    def compute_metrics(self, eval_preds):
        """Compute evaluation results for given predictions

        Args:
            eval_preds (tuple): Contains logits and labels

        Returns:
            dict: Contains precision, recall, f1 and accuracy.
        """

        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.idToLabel_[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.idToLabel_[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric_.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def postprocess(self, predictions, labels):
        """Post-process predictions- remove padding

        Args:
            predictions (Tensor): Predicted labels
            labels (Tensor): Actual labels

        Returns:
            list[list]: Actual/True labels after removing padded tokens
            list[list]: Predicted labels after removing padded tokens
        """

        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.idToLabel_[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.idToLabel_[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def local_train(self, checkpoint):
        """Perform local training for a given number of epochs

        Args:
            checkpoint: Previous model checkpoint from where training has to be started.

        Returns:
            None
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
            logger.debug(f"Root runId: {mlflow_run.data.tags.get('mlflow.rootRunId')}")
            root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")

            # log params
            self.log_params(mlflow_client, root_run_id)

            num_update_steps_per_epoch = len(self.train_loader_)
            num_training_steps = self._epochs * num_update_steps_per_epoch

            lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer_,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )

            progress_bar = tqdm(range(num_training_steps))

            for epoch in range(self._epochs):
                running_loss = 0.0
                num_of_batches_before_logging = 100
                # Training
                self.model_.train()
                for i, batch in enumerate(self.train_loader_):
                    batch = {
                        key: value.to(self.device_) for key, value in batch.items()
                    }
                    outputs = self.model_(**batch)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer_.step()
                    lr_scheduler.step()
                    self.optimizer_.zero_grad()
                    progress_bar.update(1)

                    # calculate metric
                    true_predictions, true_labels = self.postprocess(
                        outputs.logits.argmax(dim=-1), batch["labels"]
                    )
                    self.metric_.add_batch(
                        predictions=true_predictions, references=true_labels
                    )

                    running_loss += float(loss) / len(batch)

                    del outputs
                    del batch

                    if i != 0 and i % num_of_batches_before_logging == 0:
                        training_loss = running_loss / num_of_batches_before_logging
                        logger.info(
                            f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, Training Loss: {training_loss}"
                        )

                        if not self._distributed or self._rank == 0:
                            # log train loss
                            self.log_metrics(
                                mlflow_client,
                                root_run_id,
                                "Train Loss",
                                training_loss,
                            )

                            # log evaluation metrics
                            results = self.metric_.compute()
                            for key in ["precision", "recall", "f1", "accuracy"]:
                                self.log_metrics(
                                    mlflow_client,
                                    root_run_id,
                                    f"Train {key}",
                                    results[f"overall_{key}"],
                                )

                        running_loss = 0.0

                test_loss, metric_results = self.test()

                # log test loss for each epoch
                if not self._distributed or self._rank == 0:
                    self.log_metrics(mlflow_client, root_run_id, "Test Loss", test_loss)
                logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")

                # log test metric for each epoch
                for key in ["precision", "recall", "f1", "accuracy"]:
                    logger.info(
                        f"Epoch: {epoch}: Test {key} is {metric_results[f'overall_{key}']}"
                    )
                    if not self._distributed or self._rank == 0:
                        self.log_metrics(
                            mlflow_client,
                            root_run_id,
                            f"Test {key}",
                            metric_results[f"overall_{key}"],
                        )

            # log metrics for each FL iteration
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

                for key in ["precision", "recall", "f1", "accuracy"]:
                    self.log_metrics(
                        mlflow_client,
                        root_run_id,
                        f"Test {key}",
                        metric_results[f"overall_{key}"],
                        pipeline_level=True,
                    )

    def test(self):
        """Test the trained model and report test loss and metrics"""
        self.model_.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in self.test_loader_:
                batch = {key: value.to(self.device_) for key, value in batch.items()}
                outputs = self.model_(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                test_loss += float(outputs.loss)

                del outputs
                del batch

                true_predictions, true_labels = self.postprocess(predictions, labels)
                self.metric_.add_batch(
                    predictions=true_predictions, references=true_labels
                )

        results = self.metric_.compute()
        test_loss /= len(self.test_loader_)

        return test_loss, results

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
        "--tokenizer_name", type=str, required=False, help="Tokenizer model name"
    )
    parser.add_argument("--model_name", type=str, required=False, help="Model name")
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

    trainer = NERTrainer(
        tokenizer_name=args.tokenizer_name,
        model_name=args.model_name,
        train_data_dir=args.train_data,
        test_data_dir=args.test_data,
        model_path=args.model + "/model.pt",
        lr=args.lr,
        epochs=args.epochs,
        dp=args.dp,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_target_delta=args.dp_target_delta,
        dp_max_grad_norm=args.dp_max_grad_norm,
        total_num_of_iterations=args.total_num_of_iterations,
        experiment_name=args.metrics_prefix,
        iteration_num=args.iteration_num,
        batch_size=args.batch_size,
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
