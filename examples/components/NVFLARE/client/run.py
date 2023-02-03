"""This script runs an NVFlare client inside an AzureML job.

The script will:
- fetch the server IP from a pipeline root tag using mlflow,
- add server name+ip to /etc/hosts to allow client code to find server,
- run the client workspace setup script.

NOTE: the script can take an input data --client_data folder
which AzureML will mount to the job. This script passes an environment variable
to the client workspace setup script to tell it where the data is mounted.
"""
import os
import sys
import time
import logging
import argparse
import socket
import subprocess
import shutil
import tempfile
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

    group = parser.add_argument_group("NVFlare client launcher arguments")
    group.add_argument(
        "--federation_identifier",
        type=str,
        required=True,
        help="a unique identifier for the group of clients and server to find each other",
    )
    group.add_argument(
        "--client_config",
        type=str,
        required=True,
        help="the NVFlare workspace folder for this client",
    )
    group.add_argument(
        "--client_data",
        type=str,
        required=False,
        default=None,
        help="an optional folder containing data for the client to use",
    )
    group.add_argument(
        "--client_data_env_var",
        type=str,
        required=False,
        default="CLIENT_DATA_PATH",
        help="the name of the env variable to set with the mount path of the client_data folder",
    )

    return parser


def run_cli_command(cli_command: list, timeout: int = None, custom_env: dict = None):
    """Runs subprocess for a cli setup command"""
    logger = logging.getLogger()
    logger.info(f"Launching cli with command: {cli_command}")
    cli_command_call = subprocess.run(
        cli_command,
        universal_newlines=True,
        check=False,  # will not raise an exception if subprocess fails
        timeout=timeout,  # TODO: more than a minute would be weird?
        env=custom_env,
    )
    logger.info(f"return code: {cli_command_call.returncode}")

    if cli_command_call.returncode != 0:
        raise RuntimeError("Cli command returned code != 0")

    return cli_command_call.returncode


def fetch_server_ip(
    mlflow_run,
    federation_identifier,
    fetch_name=False,
    target_run_tag="mlflow.rootRunId",
    timeout=600,
):
    """Fetches the server ip and name from the mlflow run tags.

    Args:
        mlflow_run (mlflow.entities.Run): the mlflow run to fetch the tags from
        federation_identifier (str): the federation identifier to use to find the server ip
        fetch_name (bool, optional): whether to fetch the server name. Defaults to False.
        target_run_tag (str, optional): the tag to use to find the root run. Defaults to "mlflow.rootRunId".
        timeout (int, optional): the timeout in seconds to wait for the server ip. Defaults to 600.

    Returns:
        str, str: the server ip and name
    """
    logger = logging.getLogger(__name__)

    mlflow_client = mlflow.tracking.client.MlflowClient()
    logger.info(f"run tags: {mlflow_run.data.tags}")
    logger.info(
        f"target tag {target_run_tag}={mlflow_run.data.tags.get(target_run_tag)}"
    )
    target_run_id = mlflow_run.data.tags.get(target_run_tag)

    server_ip = None
    server_name = None

    fetch_start_time = time.time()

    while (fetch_name and server_name is None) or server_ip is None:
        logger.info(f"Checking out tag server_ip...")
        mlflow_root_run = mlflow_client.get_run(target_run_id)

        server_ip_tag = f"{federation_identifier}.server_ip"

        if server_ip_tag in mlflow_root_run.data.tags:
            server_ip = mlflow_root_run.data.tags[server_ip_tag]
            logger.info(f"server_ip found: {server_ip}")

        if fetch_name:
            server_name_tag = f"{federation_identifier}.server_name"

            if server_name_tag in mlflow_root_run.data.tags:
                server_name = mlflow_root_run.data.tags[server_name_tag]
                logger.info(f"server_name found: {server_name}")

        if ((fetch_name and server_name is None) or server_ip is None) and (
            time.time() - fetch_start_time > timeout
        ):
            raise RuntimeError("Could not fetch the tag within timeout.")
        else:
            time.sleep(10)

    return server_ip, server_name


def run_client(args):
    """Runs the client communication process."""
    logger = logging.getLogger()

    # use mlflow to getch server ip and name
    mlflow_run = mlflow.start_run()
    overseer_ip, overseer_name = fetch_server_ip(
        mlflow_run, args.federation_identifier, fetch_name=True
    )

    # create hosts file to resolve ip adresses
    with open("/etc/hosts", "a") as f:
        # write server address
        f.write(f"{overseer_ip}\t{overseer_name}\n")

    # create env for the client startup script
    client_env = dict(os.environ)
    if args.client_data and args.client_data_env_var:
        client_env[args.client_data_env_var] = args.client_data

    # run client startup
    client_dir_local = tempfile.TemporaryDirectory().name
    shutil.copytree(args.client_config, client_dir_local)

    startup_script_path = os.path.join(client_dir_local, "startup", "sub_start.sh")
    logger.info(f"Running ${startup_script_path}")
    run_cli_command(["bash", startup_script_path], custom_env=client_env)


def main():
    """Script main function."""
    # Create and configure logger to write into a file in job outputs/
    logging.basicConfig(
        filename="outputs/nvflare_host.log",
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        filemode="a",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("sys.argv: {}".format(sys.argv))

    # parse the arguments
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()
    logger.info("args: {}".format(args))

    run_client(args)


if __name__ == "__main__":
    main()
