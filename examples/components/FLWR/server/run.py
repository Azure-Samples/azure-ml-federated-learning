import os
import sys
import time
import logging
import argparse
import socket
import subprocess
import tempfile
import shutil
import mlflow


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

    group = parser.add_argument_group("MCD Launcher Inputs")
    group.add_argument("--federation_identifier", type=str, required=True)
    group.add_argument("--expected_clients", type=int, required=True)
    group.add_argument("--output_dir", type=str, required=True)
    group.add_argument(
        "--wait_for_clients_timeout", type=int, required=False, default=600
    )

    return parser


def publish_server_ip(
    mlflow_run, federation_identifier, target_run_tag="mlflow.rootRunId"
):
    logger = logging.getLogger()

    # get the local ip of this current node
    local_hostname = socket.gethostname()
    local_ip = socket.gethostbyname(local_hostname)
    logger.info(f"Detected IP from socket.gethostbyname(): {local_ip}")

    server_ip = str(local_ip)

    mlflow_client = mlflow.tracking.client.MlflowClient()
    logger.info(f"run tags: {mlflow_run.data.tags}")
    logger.info(
        f"target tag {target_run_tag}={mlflow_run.data.tags.get(target_run_tag)}"
    )
    target_run_id = mlflow_run.data.tags.get(target_run_tag)

    # publish the server ip as a tag in mlflow
    mlflow_client.set_tag(
        run_id=target_run_id, key=f"{federation_identifier}.server", value=server_ip
    )

    return server_ip


def run_server(
    federation_identifier,
    expected_clients,
    output_dir,
    wait_for_clients_timeout=600,
):
    """Runs the server communication process.

    Args:
        federation_identifier (str): a unique identifier for the server/clients federation
        expected_clients (int): the size of the federation
        output_dir (str): path to write the job outputs
        wait_for_clients_timeout (int): timeout in seconds to wait for clients to connect
    """
    logger = logging.getLogger()

    # communicate to clients through mlflow root (magic)
    mlflow_run = mlflow.start_run()
    server_ip = publish_server_ip(mlflow_run, federation_identifier)

    ###########################
    ### FLOWERS CODE BEGINS ###
    ###########################

    # we're intentionally just copy/pasting flowers code from https://github.com/adap/flower/blob/main/examples/quickstart_pytorch/server.py
    from typing import List, Tuple

    import flwr as fl
    from flwr.common import Metrics

    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        for i, (n, m) in enumerate(metrics):
            for k in m:
                mlflow.log_metric("client_{}_{}".format(i, k), m[k])

        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        fed_accurary = sum(accuracies) / sum(examples)
        mlflow.log_metric("fed_accurary", fed_accurary)

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": fed_accurary}

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=expected_clients,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=f"{server_ip}:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    ########################
    ### FLOWERS CODE END ###
    ########################


def main():
    """Script main function."""
    # Create and configure logger to write into a file in job outputs/
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] - %(message)s")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("sys.argv: {}".format(sys.argv))

    # parse the arguments
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()
    logger.info("args: {}".format(args))

    run_server(
        args.federation_identifier,
        args.expected_clients,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
