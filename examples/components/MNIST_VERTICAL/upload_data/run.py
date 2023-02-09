"""Script for mock components."""
import os
import sys
import argparse
import logging
import multiprocessing as mp
from functools import partial

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import mlflow

DATASET_URL = {
    "train": "https://azureopendatastorage.blob.core.windows.net/mnist/processed/train.csv",
    "test": "https://azureopendatastorage.blob.core.windows.net/mnist/processed/t10k.csv",
}


class MnistDataset(Dataset):
    """MNIST Dataset - combination of features and labels

    Args:
        feature: MNIST images tensors
        target: Tensor of labels corresponding to features
        transform: Transformation to be applied on each image

    Returns:
        None
    """

    def __init__(self, feature, target=None, transform=None):
        self.X = feature
        self.Y = target
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is None:
            if self.transform is not None:
                return self.transform(self.X[idx])
            else:
                return self.X[idx]
        if self.transform is not None:
            return self.transform(self.X[idx]), self.Y[idx]
        return self.X[idx], self.Y[idx]


def process_sample(processed_data_dir, dataset, idx):
    data = dataset.__getitem__(idx)
    output_path = processed_data_dir + f"/{idx}.jpg"
    save_image(data, output_path)


def remove_dir_contents(dir):
    import shutil

    for filename in tqdm(os.listdir(dir)):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def preprocess_data(
    raw_training_data,
    raw_testing_data,
    silo_index,
    silo_count,
    train_data_dir="./",
    test_data_dir="./",
    metrics_prefix="default-prefix",
):
    """Preprocess the raw_training_data and raw_testing_data and save the processed data to train_data_dir and test_data_dir.

    Args:
        raw_training_data: Training data directory that need to be processed
        raw_testing_data: Testing data directory that need to be processed
        silo_index: Index of the current silo
        silo_count: Number of silos in total
        train_data_dir: Train data directory where processed train data will be saved
        test_data_dir: Test data directory where processed test data will be saved
    Returns:
        None
    """

    logger.info(
        f"Raw Training Data path: {raw_training_data}, Raw Testing Data path: {raw_testing_data}, Processed Training Data dir path: {train_data_dir}, Processed Testing Data dir path: {test_data_dir}"
    )
    train_data = pd.read_csv(raw_training_data, index_col=0).reset_index()
    test_data = pd.read_csv(raw_testing_data, index_col=0).reset_index()

    # Make sure directories exist and are empty, otherwise
    # there might be images that we have not accounted for
    os.makedirs(train_data_dir, exist_ok=True)
    logger.info(f"Folder created: {train_data_dir}")
    remove_dir_contents(train_data_dir)

    os.makedirs(test_data_dir, exist_ok=True)
    logger.info(f"Folder created: {test_data_dir}")
    remove_dir_contents(test_data_dir)

    if silo_index == 0:
        train_data[["label"]].to_csv(f"{train_data_dir}/train.csv")
        test_data[["label"]].to_csv(f"{test_data_dir}/test.csv")
        return

    logger.debug(f"Segregating labels and features")
    X_train = torch.tensor(train_data.loc[:, train_data.columns != "label"].values)
    X_train = torch.reshape(X_train, (-1, 28, 28))

    X_test = torch.tensor(test_data.loc[:, test_data.columns != "label"].values)
    X_test = torch.reshape(X_test, (-1, 28, 28))

    silo_part = 28 // (silo_count - 1)
    X_train = X_train[:, (silo_index - 1) * silo_part : silo_index * silo_part, :]
    X_test = X_test[:, (silo_index - 1) * silo_part : silo_index * silo_part, :]

    X_train = X_train / 255.0
    X_test = X_test / 255.0
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=30),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )

    logger.info(
        f"Transforming images data and saving processed data to {train_data_dir} and {test_data_dir}"
    )
    train_dataset = MnistDataset(X_train.float(), transform=transform)
    test_dataset = MnistDataset(X_test.float(), transform=transform)
    datasets = {"train": train_dataset, "test": test_dataset}

    # Mlflow logging
    log_metadata(X_train, X_test, metrics_prefix)

    for x in ["train", "test"]:
        processed_data_dir = train_data_dir if x == "train" else test_data_dir

        cpu_count = max(mp.cpu_count(), 6)
        process_sample_partial = partial(
            process_sample, processed_data_dir, datasets[x]
        )

        pool = mp.Pool(processes=cpu_count)
        for _ in tqdm(
            pool.imap_unordered(process_sample_partial, range(len(datasets[x]))),
            total=len(datasets[x]),
        ):
            pass


def log_metadata(X_train, X_test, metrics_prefix):
    with mlflow.start_run() as mlflow_run:
        # get Mlflow client
        mlflow_client = mlflow.tracking.client.MlflowClient()
        logger.debug(f"Root runId: {mlflow_run.data.tags.get('mlflow.rootRunId')}")
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        mlflow_client.log_metric(
            run_id=root_run_id,
            key=f"{metrics_prefix}/Number of train datapoints",
            value=f"{X_train.size(dim=0)}",
        )

        mlflow_client.log_metric(
            run_id=root_run_id,
            key=f"{metrics_prefix}/Number of test datapoints",
            value=f"{X_test.size(dim=0)}",
        )


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    if args.silo_count < 2:
        raise Exception(
            "Number of splits/silos must be 2 and silo index must be either 0 or 1!"
        )

    preprocess_data(
        DATASET_URL["train"],
        DATASET_URL["test"],
        args.silo_index,
        args.silo_count,
        args.raw_train_data,
        args.raw_test_data,
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
        "--silo_count",
        type=int,
        required=True,
        help="Number of silos",
    )
    parser.add_argument(
        "--silo_index",
        type=int,
        required=True,
        help="Index of the current silo",
    )
    parser.add_argument(
        "--raw_train_data",
        type=str,
        required=True,
        help="Output folder for train data",
    )
    parser.add_argument(
        "--raw_test_data",
        type=str,
        required=True,
        help="Output folder for test data",
    )
    return parser


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
