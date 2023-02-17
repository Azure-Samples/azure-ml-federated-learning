# This file defining the learner was adapted from https://github.com/Azure/medical-imaging/blob/main/federated-learning/pneumonia-federated/custom/pt_learner.py to run directly on Azure ML without using NVFlare.
"""Script for training component."""
import argparse
import logging
import sys
import os.path
from distutils.util import strtobool
import time

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
TRAIN_COMPONENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../traininsilo"))
sys.path.insert(0, TRAIN_COMPONENT)
from pneumonia_network import PneumoniaNetwork

# DP
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator



def load_dataset(data_dir, transforms, benchmark_test_all_data):
    """Load dataset from {data_dir} directory. It is assumed that it contains two subdirectories 'train' and 'test'.

    Args:
        data_dir(str, optional): Data directory path
    """
    logger.info(f"Data dir: {data_dir}.")
    
    if benchmark_test_all_data:
        test_dataset = datasets.ImageFolder(
                root=os.path.join(data_dir, "all_data", "test"), transform=transforms)
    else:
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "test"), transform=transforms )


    return  test_dataset


def log_metrics(client, run_id, key, value, pipeline_level=False):
    if run_id:
        client.log_metric(
            run_id=run_id,
            key=f"evaluation/{key}",
            value=value,
        )



def test(args, device_id, distributed):
    """Test the aggregated model and report test loss and accuracy"""

    device_ = (
        torch.device(
            torch.device("cuda", device_id) if torch.cuda.is_available() else "cpu"
        )
        if device_id is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device_}")

    if distributed:
        _rank = device_id
        logger.info(f"Rank: {_rank}")
    else:
        _rank = None

    # Training setup
    model_path = os.path.join(args.checkpoint, "model.pt")
    model_ = PneumoniaNetwork()
    model_.to(device_)
    if distributed:
        model_ = DDP(
            model_,
            device_ids=[_rank] if _rank is not None else None,
            output_device=_rank,
        )
    if distributed:
        # DDP comes with "module." prefix: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        model_.module.load_state_dict(
            torch.load(model_path, map_location=device_)
        )
    else:
        model_.load_state_dict(
            torch.load(model_path, map_location=device_)
        )

    loss_ = nn.CrossEntropyLoss()

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
    test_dataset_ = load_dataset(
        data_dir=args.test_data_dir, transforms=transforms, benchmark_test_all_data=args.benchmark_test_all_data
    )

    if distributed:
        logger.info("Setting up distributed samplers.")
        test_sampler_ = DistributedSampler(test_dataset_)
    else:
        test_sampler_ = None

    test_loader_ = DataLoader(
        dataset=test_dataset_,
        batch_size=100,
        shuffle=False,
        sampler=test_sampler_,
    )
    logger.info(f"Test loader steps: {len(test_loader_)}")
    model_.eval()
    test_loss = 0
    correct = 0

    with mlflow.start_run() as mlflow_run:
        mlflow_client = mlflow.tracking.client.MlflowClient()
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        logger.debug(f"Root runId: {root_run_id}")
        with torch.no_grad():
            for data, target in test_loader_:
                data, target = data.to(device_), target.to(device_)
                output = model_(data)
                test_loss += loss_(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader_.dataset)
        acc = correct / len(test_loader_.dataset)

        if not distributed or _rank == 0:
            log_metrics(
                mlflow_client,
                root_run_id,
                "Test Loss",
                test_loss,
                pipeline_level=True,
            )
            log_metrics(
                mlflow_client,
                root_run_id,
                "Test Accuracy",
                acc,
                pipeline_level=True,
            )
            logger.info(
                    f"Test loss:{test_loss}"
                )
            logger.info(
                    f"Test accuracy:{acc}"
                )




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
        "--predictions_path", type=str, required=True, help="Path to save the final predictions"
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

    if int(os.environ["WORLD_SIZE"]) > 1 and torch.cuda.is_available():
        dist.init_process_group(
            "nccl",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
    elif int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("gloo")

    device_id = int(os.environ["RANK"]) if int(os.environ["WORLD_SIZE"]) > 1 else None
    distributed = int(os.environ["WORLD_SIZE"]) >1
    test(args, device_id, distributed )

    if int(os.environ["WORLD_SIZE"]) > 1:
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
