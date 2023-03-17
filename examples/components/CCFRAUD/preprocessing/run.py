import os
import argparse
import logging
import sys
import numpy as np

from sklearn.preprocessing import StandardScaler
import pandas as pd
import mlflow

# helper with confidentiality
from confidential_io import EncryptedFile

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


def apply_transforms(df):
    """Applies transformation for datetime and numerical columns

    Args:
        df (pd.DataFrame):
        dataframe to transform

    Returns:
        pd.DataFrame: transformed dataframe
    """
    global SCALERS

    datetimes = ["trans_date_trans_time"]  # "dob"
    normalize = [
        "age",
        "merch_lat",
        "merch_long",
        "lat",
        "long",
        "city_pop",
        "trans_date_trans_time",
        "amt",
    ]

    for column in datetimes:
        df.loc[:, column] = pd.to_datetime(df[column]).view("int64")
    for column in normalize:
        if column not in SCALERS:
            print(f"Creating encoder for column: {column}")
            # Simply set all zeros if the category is unseen
            scaler = StandardScaler()
            scaler.fit(df[column].values.reshape(-1, 1))
            SCALERS[column] = scaler

        scaler = SCALERS.get(column)
        df.loc[:, column] = scaler.transform(df[column].values.reshape(-1, 1))

    return df


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
    with EncryptedFile(raw_training_data + f"/train.csv", mode="rt") as train_f:
        train_df = pd.read_csv(train_f)
    with EncryptedFile(raw_testing_data + f"/test.csv", mode="rt") as test_f:
        test_df = pd.read_csv(test_f)

    fraud_weight = (
        train_df["is_fraud"].value_counts()[0] / train_df["is_fraud"].value_counts()[1]
    )
    logger.debug(f"Fraud weight: {fraud_weight}")

    logger.debug(f"Applying transformations...")
    train_data = apply_transforms(train_df)
    test_data = apply_transforms(test_df)

    logger.debug(f"Train data samples: {len(train_data)}")
    logger.debug(f"Test data samples: {len(test_data)}")

    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    train_data = train_data.sort_values(by="trans_date_trans_time")
    test_data = test_data.sort_values(by="trans_date_trans_time")

    logger.info(f"Saving processed data to {train_data_dir} and {test_data_dir}")
    with EncryptedFile(train_data_dir + "/data.csv", mode="wt") as train_f:
        train_data.to_csv(train_f, index=False)
    with EncryptedFile(test_data_dir + "/data.csv", mode="wt") as test_f:
        test_data.to_csv(test_f, index=False)
    with EncryptedFile(train_data_dir + "/fraud_weight.txt", mode="wt") as fraud_f:
        np.savetxt(fraud_f, np.array([fraud_weight]))

    # Mlflow logging
    log_metadata(train_data, test_data, metrics_prefix)


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
