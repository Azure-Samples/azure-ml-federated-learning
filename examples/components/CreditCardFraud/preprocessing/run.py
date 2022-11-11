import os
import argparse
import logging
import sys
import json

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import mlflow
import hydra

from azureml.core import Run, Workspace

ENCODERS = {}
STATES_REGIONS = {}

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
    global ENCODERS

    useful_props = [
        "amt",
        "cc_num",
        "merch_lat",
        "merch_long",
        "category",
        "region",
        "gender",
        "state",
        "zip",
        "lat",
        "long",
        "city_pop",
        "job",
        "dob",
        "trans_date_trans_time",
        "is_fraud",
    ]
    categorical = ["category", "region", "gender", "state", "job"]
    datetimes = ["dob", "trans_date_trans_time"]
    normalize = ["dob", "age"]

    # Filter only useful columns
    df = df[useful_props]

    df.loc[:, "age"] = (pd.Timestamp.now() - pd.to_datetime(df["dob"])) // pd.Timedelta('1y')
    for column in categorical:
        if column not in ENCODERS:
            print(f"Creating encoder for column: {column}")
            # Simply set all zeros if the category is unseen
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(df[column].values.reshape(-1,1))
            ENCODERS[column] = encoder

        encoder = ENCODERS.get(column)
        encoded_data = encoder.transform(df[column].values.reshape(-1,1)).toarray()
        encoded_df = pd.DataFrame(encoded_data, columns = [column + "_" + "_".join(x.split("_")[1:]) for x in encoder.get_feature_names()])
        encoded_df.index = df.index
        df = df.join(encoded_df).drop(column, axis=1)

    for column in datetimes:
        df.loc[:, column] = pd.to_datetime(df[column]).view("int64")
    for column in normalize:
        df.loc[:, column] = (df[column] - df[column].min())/(df[column].max() - df[column].min())

    return df


def preprocess_data(
    raw_training_data,
    raw_testing_data,
    train_data_dir="./",
    test_data_dir="./",
    metrics_prefix="default-prefix",
    config=None,
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

    logger.debug("Loading regions from \"us_regions.csv\"")
    global STATES_REGIONS
    df_states_regions = pd.read_csv(hydra.utils.get_original_cwd() + "/us_regions.csv")
    STATES_REGIONS = {
        row.StateCode: row.Region for row in df_states_regions.itertuples()
    }
    REGIONS = list(df_states_regions["Region"].unique())


    logger.debug(f"Loading data...")
    train_data = pd.read_csv(raw_training_data)
    test_data = pd.read_csv(raw_testing_data)

    logger.debug(f"Filtering regions...")
    train_data.loc[:, "region"] = train_data["state"].map(STATES_REGIONS)
    train_data = train_data[train_data["region"].str.match(config.region)]
    test_data.loc[:, "region"] = test_data["state"].map(STATES_REGIONS)
    test_data = test_data[test_data["region"].str.match(config.region)]

    logger.debug(f"Applying transformations...")
    train_data = apply_transforms(train_data)
    test_data = apply_transforms(test_data)

    logger.info(
        f"Saving processed data to {train_data_dir} and {test_data_dir}"
    )
    train_data.to_csv(train_data_dir + "/data.csv")
    test_data.to_csv(test_data_dir + "/data.csv")

    # Mlflow logging
    log_metadata(train_data, test_data, metrics_prefix)


def log_metadata(train_df, test_df, metrics_prefix):
    with mlflow.start_run() as mlflow_run:
        # get Mlflow client
        mlflow_client = mlflow.tracking.client.MlflowClient()
        logger.debug(f"Root runId: {mlflow_run.data.tags.get('mlflow.rootRunId')}")
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
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
    print(f"Running script with arguments: {args}")

    # Get runtime specific configuration
    run: Run = Run.get_context()
    compute_target = run.get_details()['target']
    print(f"Compute target: {compute_target}")
    sys.argv = [f"--config_name={compute_target}"]


    @hydra.main(config_path="config", config_name="default")
    def run(runtime_config):
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
            config=runtime_config,
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