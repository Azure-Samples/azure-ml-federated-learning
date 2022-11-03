import os
import argparse
import logging
import sys
import json

from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
import torch
import pandas as pd
import mlflow
import multiprocessing as mt
from functools import partial


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

    parser.add_argument("--raw_training_data", type=str, required=True, help="")
    parser.add_argument("--raw_testing_data", type=str, required=True, help="")
    parser.add_argument("--train_output", type=str, required=True, help="")
    parser.add_argument("--test_output", type=str, required=True, help="")
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    return parser


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
        if self.transform is not None:
            return self.transform(self.X[idx]), self.Y[idx]
        elif self.Y is None:
            return [self.X[idx]]
        return self.X[idx], self.Y[idx]

def process_sample(processed_data_dir, idxdatatarget):
    idx, (data, target) = idxdatatarget

    os.makedirs(processed_data_dir + f"/{target}", exist_ok=True)

    output_path = processed_data_dir + f"/{target}/{idx}.jpg"
    save_image(data, output_path)

    json_line_sample = {
        "image_url": f"./{target}/{idx}.jpg",
        # "image_url": f"azureml://subscriptions/<my-subscription-id>/resourcegroups/<my-resource-group>/workspaces/<my-workspace>/datastores/<my-datastore>/paths/<path_to_image>",
        "image_details": {"format": "jpg", "width": data.shape[1], "height": data.shape[0]},
        "label": str(target.item()),
    }
    return json.dumps(json_line_sample)

def preprocess_data(
    raw_training_data,
    raw_testing_data,
    train_data_dir="./",
    test_data_dir="./",
    metrics_prefix="default-prefix",
):
    """Preprocess the raw_training_data and raw_testing_data and save the processed data to train_data_dir and test_data_dir.

    Args:
        raw_training_data: Training data directory that need to be processed
        raw_testing_data: Testing data directory that need to be processed
        train_data_dir: Train data directory where processed train data will be saved
        test_data_dir: Test data directory where processed test data will be saved
    Returns:
        None
    """

    logger.info(
        f"Raw Training Data path: {raw_training_data}, Raw Testing Data path: {raw_testing_data}, Processed Training Data dir path: {train_data_dir}, Processed Testing Data dir path: {test_data_dir}"
    )
    train_data = pd.read_csv(raw_training_data)
    test_data = pd.read_csv(raw_testing_data)

    logger.debug(f"Segregating labels and features")
    X_train = torch.tensor(train_data.loc[:, train_data.columns != "label"].values)
    X_train = torch.reshape(X_train, (-1, 28, 28))
    y_train = torch.tensor(train_data["label"].values)

    X_test = torch.tensor(test_data.loc[:, test_data.columns != "label"].values)
    X_test = torch.reshape(X_test, (-1, 28, 28))
    y_test = torch.tensor(test_data["label"].values)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    train_set = MnistDataset(X_train.float(), y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    data = next(iter(train_loader))
    mean = data[0].mean()
    std = data[0].std()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=30),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    logger.info(
        f"Transforming images data and saving processed data to {train_data_dir} and {test_data_dir}"
    )
    train_dataset = MnistDataset(X_train.float(), y_train, transform=transform)
    test_dataset = MnistDataset(X_test.float(), y_test, transform=transform)
    datasets = {"train": train_dataset, "test": test_dataset}

    # Mlflow logging
    log_metadata(X_train, X_test, metrics_prefix)

    for x in ["train", "test"]:
        processed_data_dir = train_data_dir if x == "train" else test_data_dir
        jsonl_file_path = processed_data_dir + f"/{x}_annotations.jsonl"
        os.makedirs(processed_data_dir, exist_ok=True)

        cpu_count = mt.cpu_count()
        process_sample_partial = partial(process_sample, processed_data_dir)
        with mt.Pool(cpu_count) as pool:
            jsonl_file_contents = list(tqdm(
                pool.imap(process_sample_partial, enumerate(datasets[x])), 
                total=len(datasets[x])
            ))

        with open(jsonl_file_path, "w") as jsonl_file:
            jsonl_file.write("\n".join(jsonl_file_contents))

        MLTable_filecontent = \
f"""paths:
  - file: ./{x}_annotations.jsonl
transformations:
  - read_json_lines:
        encoding: utf8
        invalid_lines: error
        include_path_column: false
  - convert_column_types:
      - columns: image_url
        column_type: stream_info"""

        MLTable_file = processed_data_dir + f"/MLTable"
        with open(MLTable_file, "w") as f:
            f.write(MLTable_filecontent)


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

    preprocess_data(
        args.raw_training_data,
        args.raw_testing_data,
        args.train_output,
        args.test_output,
        args.metrics_prefix,
    )


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
