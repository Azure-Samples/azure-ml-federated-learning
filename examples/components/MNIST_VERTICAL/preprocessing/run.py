import os
import argparse
import logging
import sys
import numpy as np

from sklearn.preprocessing import StandardScaler
import pandas as pd
import mlflow

SCALERS = {}


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

    logger.debug(f"Loading data...")
    if os.path.exists(raw_training_data + f"/train.csv") and os.path.exists(
        raw_testing_data + f"/test.csv"
    ):
        train_df = pd.read_csv(raw_training_data + f"/train.csv", index_col=0)
        test_df = pd.read_csv(raw_testing_data + f"/test.csv", index_col=0)

        # Add matching id column
        train_df["matching_id"] = train_df.index
        test_df["matching_id"] = test_df.index
    else:
        train_df = pd.DataFrame(
            {
                "matching_id": [
                    ".".join(file.split(".")[:-1])
                    for file in os.listdir(raw_training_data)
                ]
            }
        )
        test_df = pd.DataFrame(
            {
                "matching_id": [
                    ".".join(file.split(".")[:-1])
                    for file in os.listdir(raw_testing_data)
                ]
            }
        )

    # Shuffle data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # Drop 5% of samples
    train_df = train_df.sample(frac=0.95).reset_index(drop=True)
    test_df = test_df.sample(frac=0.95).reset_index(drop=True)

    train_df.to_csv(train_data_dir + "/train_processed.csv", index=False)
    test_df.to_csv(test_data_dir + "/test_processed.csv", index=False)

    # Mlflow logging
    log_metadata(train_df, test_df, metrics_prefix)


def log_metadata(train_df, test_df, metrics_prefix):
    with mlflow.start_run() as mlflow_run:
        # get Mlflow client
        mlflow_client = mlflow.tracking.client.MlflowClient()
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        logger.debug(f"Root runId: {root_run_id}")
        if root_run_id:
            mlflow_client.log_metric(
                run_id=root_run_id,
                key=f"{metrics_prefix}/Number of train datapoints",
                value=f"{train_df.shape[0]}",
            )

            mlflow_client.log_metric(
                run_id=root_run_id,
                key=f"{metrics_prefix}/Number of test datapoints",
                value=f"{test_df.shape[0]}",
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
    logger.info(f"Running script with arguments: {args}")

    def run():
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

    run()


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
