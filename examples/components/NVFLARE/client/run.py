import os
import sys
import time
import logging
import argparse
import socket
import subprocess
import shutil
import tempfile
from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import (
    APISyntaxError,
    FLAdminAPIResponse,
    FLAdminAPISpec,
    TargetType,
)
import mlflow
from nvflare.security.logging import secure_format_exception


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
    group.add_argument("--client_config", type=str, required=True)
    group.add_argument("--client_data", type=str, required=False, default=None)
    group.add_argument(
        "--client_data_env_var", type=str, required=False, default="CLIENT_DATA_PATH"
    )

    return parser


def run_cli_command(cli_command: list, timeout: int = None, custom_env: dict = None):
    """Runs subprocess for a cli setup command"""
    logger = logging.getLogger()
    logger.info(f"Launching cli with command: {cli_command}")
    cli_command_call = subprocess.run(
        cli_command,
        # stdout=PIPE,
        # stderr=PIPE,
        universal_newlines=True,
        check=False,  # will not raise an exception if subprocess fails
        timeout=timeout,  # TODO: more than a minute would be weird?
        env=custom_env,
    )
    logger.info(f"return code: {cli_command_call.returncode}")

    if cli_command_call.returncode != 0:
        raise RuntimeError("Cli command returned code != 0")

    return cli_command_call.returncode


def run_client(client_dir):
    """Runs the client communication process."""
    logger = logging.getLogger()

    # get Mlflow client and root run id
    mlflow_run = mlflow.start_run()
    mlflow_client = mlflow.tracking.client.MlflowClient()
    logger.debug(f"parent runId: {mlflow_run.data.tags.get('mlflow.parentRunId')}")
    root_run_id = mlflow_run.data.tags.get("mlflow.parentRunId")

    overseer_name = None
    overseer_ip = None
    fetch_start_time = time.time()

    client_dir_local = tempfile.TemporaryDirectory().name
    shutil.copytree(client_dir, client_dir_local)

    while overseer_name is None or overseer_ip is None:
        logger.info(f"Checking out tag overseer_name/overseer_ip...")
        mlflow_root_run = mlflow_client.get_run(root_run_id)

        if "overseer_name" in mlflow_root_run.data.tags:
            overseer_name = mlflow_root_run.data.tags["overseer_name"]
            logger.info(f"overseer_name found: {overseer_name}")

        if "overseer_ip" in mlflow_root_run.data.tags:
            overseer_ip = mlflow_root_run.data.tags["overseer_ip"]
            logger.info(f"overseer_ip found: {overseer_ip}")

        if (overseer_name is None or overseer_ip is None) and (
            time.time() - fetch_start_time > 600
        ):
            raise RuntimeError("Could not fetch the tag within timeout.")
        else:
            time.sleep(10)

    # create hosts file to resolve ip adresses
    with (open("/etc/hosts", "a")) as f:
        # write server address
        f.write(f"{overseer_ip}\t{overseer_name}\n")

    # create env for the client startup script
    client_env = dict(os.environ)
    if args.client_data and args.client_data_env_var:
        client_env[args.client_data_env_var] = args.client_data

    # run client startup
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

    run_client(args.client_config)


if __name__ == "__main__":
    main()
