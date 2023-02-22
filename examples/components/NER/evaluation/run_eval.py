import argparse
import logging
import sys
import os

from transformers import (
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk
from torch.optim import AdamW
import pandas as pd
import evaluate
import mlflow
import torch
from distutils.util import strtobool
import multiprocessing
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



def load_dataset(test_data_dir):
    """Load dataset from {train_data_dir} and {test_data_dir}

    Args:
        train_data_dir(str, optional): Training data directory path
        test_data_dir(str, optional): Testing data directory path

    Returns:
        Train dataset
        Test datset
    """
    logger.info(f"Test data dir: {test_data_dir}")
    test_dataset = load_from_disk(test_data_dir)

    return  test_dataset


def log_metrics( client, run_id, key, value):
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
        client.log_metric(
            run_id=run_id,
            key=f"Evaluation/{key}",
            value=value,
        )

def postprocess(predictions, labels, idToLabel_):
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
        [idToLabel_[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [idToLabel_[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


def test(args, device_id, _distributed):
    """Test the trained model and report test loss and metrics"""
    # get Mlflow client and root run id
    device_ = (
        torch.device(
            torch.device("cuda", device_id) if torch.cuda.is_available() else "cpu"
        )
        if device_id is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device_}")

    if _distributed:
        _rank = device_id
        logger.info(f"Rank: {_rank}")
    else:
        _rank = None

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # dataset and data loader
    data_collator = DataCollatorForPrivateTokenClassification(tokenizer=tokenizer)
    partial_test_path = os.path.join(args.test_data_dir, "partial_test")
    test_dataset = load_dataset(partial_test_path)

    # load all train and test if run benchmark
    if args.benchmark_test_all_data:

        all_test_path = os.path.join(args.test_data_dir, "all_test")
        test_dataset = load_dataset( all_test_path)

    if _distributed:
        logger.info("Setting up distributed samplers.")
        test_sampler_ = DistributedSampler(test_dataset)
    else:
        test_sampler_ = None

    #get number of cpu to load data for each gpu
    num_workers_per_gpu = int(multiprocessing.cpu_count()//int(os.environ['WORLD_SIZE']))
    logger.info(f"The num_work per GPU is: {num_workers_per_gpu}")

    test_loader_ = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        sampler=test_sampler_,
    )

    logger.info(f"Test loader steps: {len(test_loader_)}")

    # training params
    labelToId_ = pd.read_json("./labels.json", typ="series").to_dict()
    idToLabel_ = {val: key for key, val in labelToId_.items()}
    model_ = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        id2label=idToLabel_,
        label2id=labelToId_,
    )
    trainable_layers = [model_.bert.encoder.layer[-1], model_.classifier]
    trainable_params = 0
    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()
    logger.info(f"Trainable parameters: {trainable_params}")

    model_.to(device_)
    if _distributed:
        model_ = DDP(
            model_,
            device_ids=[_rank] if _rank is not None else None,
            output_device=_rank,
        )
    metric_ = evaluate.load("seqeval")

    optimizer_ = AdamW(model_.parameters(), lr=2e-5)

    

    if _distributed:
        # DDP comes with "module." prefix: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        model_.module.load_state_dict(
            torch.load(args.checkpoint + "/model.pt", map_location=device_)
        )
    else:
        model_.load_state_dict(
            torch.load(args.checkpoint + "/model.pt", map_location=device_)
        )
    with mlflow.start_run() as mlflow_run:
        mlflow_client = mlflow.tracking.client.MlflowClient()
        logger.debug(f"Root runId: {mlflow_run.data.tags.get('mlflow.rootRunId')}")
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        model_.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader_:
                batch = {key: value.to(device_) for key, value in batch.items()}
                outputs = model_(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                test_loss += float(outputs.loss)

                del outputs
                del batch

                true_predictions, true_labels = postprocess(predictions, labels, idToLabel_)
                metric_.add_batch(
                    predictions=true_predictions, references=true_labels
                )

        metric_results = metric_.compute()
        test_loss /= len(test_loader_)

        # log metrics for each FL iteration
        if not _distributed or _rank == 0:
            log_metrics(
                mlflow_client,
                root_run_id,
                "Test Loss",
                test_loss,
            )

            for key in ["precision", "recall", "f1", "accuracy"]:
                log_metrics(
                    mlflow_client,
                    root_run_id,
                    f"Test {key}",
                    metric_results[f"overall_{key}"],
                )
            logger.info(f"Test loss: {test_loss}")
            for key in ["precision", "recall", "f1", "accuracy"]:
                logger.info(f"Test {key}: {metric_results[f'overall_{key}']}")
                

    



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
        "--test_data_dir", type=str, required=True, help="Path to input test data"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help=""
    )
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, help="Batch Size"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Class name of the model"
    )
    parser.add_argument(
        "--predictions_path", type=str, required=True, help="Path to save the final predictions"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, required=False, help="Tokenizer model name"
    )
    parser.add_argument(
        "--benchmark_test_all_data", type=strtobool, required=False,help="Whether to use all test data (all silos combined) to bechmark final aggregated model"
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

    
    test(args, device_id=int(os.environ['RANK']), _distributed=int(os.environ['WORLD_SIZE']) > 1)

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
