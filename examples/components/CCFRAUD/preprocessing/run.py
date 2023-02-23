import os
import argparse
import logging
import sys
import numpy as np

from sklearn.preprocessing import StandardScaler
import pandas as pd
import mlflow
from distutils.util import strtobool

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
    parser.add_argument(
        "--benchmark_test_all_data", type=strtobool, required=False,help="Whether to use all test data (all silos combined) to bechmark final aggregated model"
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
    benchmark_test_all_data,
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

    ########## DDP benchmark############
    
    for i in range(10):
        if i ==0:
            train_df = pd.read_csv(raw_training_data + f"/train_filtered_0.csv")
        else:  
            train_df = train_df.append(pd.read_csv(raw_training_data + f"/train_filtered_{i}.csv"))

    for i in range(10):
        if i ==0:
            test_df = pd.read_csv(raw_testing_data + f"/test_filtered_0.csv")
        else:  
            test_df = test_df.append(pd.read_csv(raw_testing_data + f"/test_filtered_{i}.csv"))
    

    #train_df = pd.read_csv(raw_training_data + f"/train_filtered.csv")
    #test_df = pd.read_csv(raw_testing_data + f"/test_filtered.csv")

    fraud_weight = (
        train_df["is_fraud"].value_counts()[0] / train_df["is_fraud"].value_counts()[1]
    )
    logger.debug(f"Fraud weight: {fraud_weight}")

    logger.debug(f"Applying transformations...")
    train_data = apply_transforms(train_df)
    test_data = apply_transforms(test_df)

    
    logger.debug(f"Train data samples: {len(train_data)}")
    logger.debug(f"Test data samples: {len(test_data)}")

    train_data = train_data.sort_values(by="trans_date_trans_time")
    test_data = test_data.sort_values(by="trans_date_trans_time")

    train_data.to_csv(train_data_dir + "/filtered_data.csv", index=False)
    np.savetxt(train_data_dir + "/filtered_fraud_weight.txt", np.array([fraud_weight]))
    test_data.to_csv(test_data_dir + "/filtered_data.csv", index=False)

    # Mlflow logging
    log_metadata(train_data, test_data, metrics_prefix, "filtered")

    
    # if run benchmark, process all data too
    if benchmark_test_all_data:

        train_df_unfiltered  = pd.read_csv(raw_training_data + f"/train_unfiltered.csv")
        test_df_unfiltered = pd.read_csv(raw_testing_data + f"/test_unfiltered.csv")
        fraud_weight_unfiltered = (
            train_df_unfiltered["is_fraud"].value_counts()[0] / train_df_unfiltered["is_fraud"].value_counts()[1]
        )
        logger.debug(f"Fraud weight unfiltered: {fraud_weight_unfiltered}")
        train_data_unfiltered = apply_transforms(train_df_unfiltered)
        test_data_unfiltered = apply_transforms(test_df_unfiltered)
        logger.debug(f"Train data samples unfiltered: {len(train_data_unfiltered)}")
        logger.debug(f"Test data samples unfiltered: {len(test_data_unfiltered)}")
        train_data_unfiltered = train_data_unfiltered.sort_values(by="trans_date_trans_time")
        test_data_unfiltered = test_data_unfiltered.sort_values(by="trans_date_trans_time")
        train_data_unfiltered.to_csv(train_data_dir + "/unfiltered_data.csv", index=False)
        np.savetxt(train_data_dir + "/unfiltered_fraud_weight.txt", np.array([fraud_weight_unfiltered]))
        test_data_unfiltered.to_csv(test_data_dir + "/unfiltered_data.csv", index=False)

        log_metadata(train_data_unfiltered, test_data_unfiltered, metrics_prefix, "unfiltered")

    # Mlflow logging
    logger.info(f"Saving processed data to {train_data_dir} and {test_data_dir}")
    

    
    


def log_metadata(train_df, test_df, metrics_prefix, property):
    with mlflow.start_run() as mlflow_run:
        # get Mlflow client
        mlflow_client = mlflow.tracking.client.MlflowClient()
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        logger.debug(f"Root runId: {root_run_id}")
        if root_run_id:
            mlflow_client.log_metric(
                run_id=root_run_id,
                key=f"{metrics_prefix}/Number of {property} train datapoints",
                value=f"{train_df.shape[0]}",
            )

            mlflow_client.log_metric(
                run_id=root_run_id,
                key=f"{metrics_prefix}/Number of {property} test datapoints",
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
            args.benchmark_test_all_data,
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
