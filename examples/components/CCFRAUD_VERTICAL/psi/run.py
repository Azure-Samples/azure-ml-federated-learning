import os
import argparse
import logging
import sys
import shutil
from distutils.util import strtobool

from aml_comm import AMLCommSocket, AMLCommRedis
from aml_smpc import AMLSMPC
import pandas as pd
import mlflow

# Find more about APSI library here: https://github.com/microsoft/APSI
from SymmetricPSI import PSISender, PSIReceiver


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
        "--matching_column_name", type=str, required=False, default=None, help=""
    )
    parser.add_argument(
        "--global_size",
        type=int,
        required=True,
        help="Number of silos",
    )
    parser.add_argument(
        "--global_rank",
        type=int,
        required=True,
        help="Index of the current silo",
    )
    parser.add_argument(
        "--communication_backend",
        type=str,
        required=False,
        default="socket",
        help="Type of communication to use between the nodes",
    )
    parser.add_argument(
        "--communication_encrypted",
        type=strtobool,
        required=False,
        default=False,
        help="Encrypt messages exchanged between the nodes",
    )
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    return parser


def run_psi(df, matching_column_name, rank, world_size, comm):
    if matching_column_name is None:
        local_ids = list(df.index.astype(str))
    else:
        local_ids = list(df[matching_column_name].astype(str))

    if rank == 0:
        psi_receiver = PSIReceiver(local_ids)
        psi_request = psi_receiver.create_request()
        for i in range(1, world_size):
            comm.send(psi_request, i)

        filters = [None] * world_size
        for i in range(1, world_size):
            filters[i] = comm.recv(i)

        responses = [None] * world_size
        for i in range(1, world_size):
            responses[i] = comm.recv(i)

        overlaps = [None] * world_size
        for i in range(1, world_size):
            overlaps[i] = psi_receiver.find_overlap(filters[i], responses[i])

        overlap = set(local_ids)
        for i in range(1, world_size):
            overlap = overlap.intersection(overlaps[i])

        for i in range(1, world_size):
            comm.send(overlap, i)
    else:
        psi_sender = PSISender(local_ids)
        psi_request = comm.recv(0)
        filter = psi_sender.get_serialized_filter()
        comm.send(filter, 0)
        response = psi_sender.create_request_response(psi_request)
        comm.send(response, 0)
        overlap = comm.recv(0)
    return overlap


def psi(
    comm,
    rank,
    world_size,
    raw_training_data,
    raw_testing_data,
    matching_column_name,
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
    train_df = pd.read_csv(raw_training_data + f"/data.csv", index_col=0)
    test_df = pd.read_csv(raw_testing_data + f"/data.csv", index_col=0)

    logger.debug(f"Train data shape before PSI: {train_df.shape}")
    overlap = run_psi(train_df, matching_column_name, rank, world_size, comm)
    if matching_column_name is not None:
        train_df = train_df[train_df[matching_column_name].astype(str).isin(overlap)]
    else:
        train_df = train_df[train_df.index.astype(str).isin(overlap)]
    logger.debug(f"Train data shape after PSI: {train_df.shape}")

    logger.debug(f"Test data shape before PSI: {test_df.shape}")
    overlap = run_psi(test_df, matching_column_name, rank, world_size, comm)
    if matching_column_name is not None:
        test_df = test_df[test_df[matching_column_name].astype(str).isin(overlap)]
    else:
        test_df = test_df[test_df.index.astype(str).isin(overlap)]
    logger.debug(f"Test data shape after PSI: {test_df.shape}")

    train_df.to_csv(train_data_dir + "/data.csv")
    test_df.to_csv(test_data_dir + "/data.csv")

    if rank == 0:
        shutil.copy(
            raw_training_data + "/fraud_weight.txt",
            train_data_dir + "/fraud_weight.txt",
        )

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

    if args.communication_encrypted:
        encryption = AMLSMPC()
    else:
        encryption = None

    if args.communication_backend == "socket":
        global_comm = AMLCommSocket(
            args.global_rank,
            args.global_size,
            os.environ.get("AZUREML_ROOT_RUN_ID"),
            encryption=encryption,
        )
    elif args.communication_backend == "redis":
        global_comm = AMLCommRedis(
            args.global_rank,
            args.global_size,
            os.environ.get("AZUREML_ROOT_RUN_ID"),
            encryption=encryption,
        )
    else:
        raise ValueError("Communication backend not supported")

    psi(
        global_comm,
        args.global_rank,
        args.global_size,
        args.raw_training_data,
        args.raw_testing_data,
        args.matching_column_name,
        args.train_output,
        args.test_output,
        args.metrics_prefix,
    )


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
