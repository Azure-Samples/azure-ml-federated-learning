import os
import argparse
import logging
import sys
import socket

from aml_comm import AMLCommSocket
import pandas as pd
import numpy as np
import mlflow

from SymmetricPSI import PSISender, PSIReceiver

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
        "--rank",
        type=int,
        required=True,
        help="Index of the current silo",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="Index of the current silo",
    )
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    return parser


def psi(
    rank,
    world_size,
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

    comm = AMLCommSocket(rank, world_size, os.environ.get("AZUREML_ROOT_RUN_ID"))

    logger.info(
        f"Raw Training Data path: {raw_training_data}, Raw Testing Data path: {raw_testing_data}, Processed Training Data dir path: {train_data_dir}, Processed Testing Data dir path: {test_data_dir}"
    )

    logger.debug(f"Loading data...")
    train_df = pd.read_csv(raw_training_data + f"/train_processed.csv")
    test_df = pd.read_csv(raw_testing_data + f"/test_processed.csv")

    logger.debug(f"Train data shape: {train_df.shape}")
    if rank == 0:
        local_ids = list(train_df["matching_id"])
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

        train_df = train_df[train_df["matching_id"].isin(overlap)]
    else:
        psi_sender = PSISender()
        psi_request = comm.recv(0)
        filter = psi_sender.get_serialized_filter()
        comm.send(filter, 0)
        response = psi_sender.create_request_response(psi_request)
        comm.send(response, 0)
        overlap = comm.recv(0)
        train_df = train_df[train_df["matching_id"].isin(overlap)]

    logger.debug(f"Train data shape after PSI: {train_df.shape}")

    # train_df[["matching_id"]].to_csv(
    #     "./train_matching_ids.csv", index=False, header=False
    # )
    # test_df[["matching_id"]].to_csv("./test_matching_ids.csv", index=True, header=False)

    # # Execute command in shell
    # if rank == 0:
    #     for i in range(1, world_size):
    #         logger.debug(f"Receiving ip address from rank {i}")
    #         ip_address = comm.recv(i)
    #         logger.debug(f"Received ip address {ip_address} from rank {i}")

    #         os.system(
    #             f"/APSI/build/bin/receiver_cli -l all -q ./train_matching_ids.csv -o ./train_matched_ids_{i}.csv -a {ip_address}"
    #         )
    #         os.system(
    #             f"/APSI/build/bin/receiver_cli -l all -q ./test_matching_ids.csv -o ./test_matched_ids_{i}.csv -a {ip_address}"
    #         )

    #         logger.debug(
    #             f"Matched ids from rank {i} are saved to ./train_matched_ids_{i}.csv and ./test_matched_ids_{i}.csv"
    #         )
    #         logger.debug(
    #             f"Head of train_matched_ids_{i}.csv: {pd.read_csv(f'./train_matched_ids_{i}.csv').head()}"
    #         )
    # else:
    #     ip_address = socket.gethostbyname(socket.gethostname())
    #     logger.debug(f"Sending ip address {ip_address} to rank 0")
    #     comm.send(ip_address, 0)

    #     os.system(
    #         "/APSI/build/bin/sender_cli -l all -d ./train_matching_ids.csv -p ./params.json"
    #     )
    #     os.system(
    #         "/APSI/build/bin/sender_cli -l all -d ./test_matching_ids.csv -p ./params.json"
    #     )

    train_df.to_csv(train_data_dir + "/data.csv")
    test_df.to_csv(test_data_dir + "/data.csv")

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

        psi(
            args.rank,
            args.world_size,
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
