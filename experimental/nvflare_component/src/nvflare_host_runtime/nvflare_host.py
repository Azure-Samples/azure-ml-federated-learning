import os
import sys
import time
import logging
import argparse
import socket
import subprocess
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

    group = parser.add_argument_group("MCD Launcher Inputs")
    group.add_argument("--run_id", type=str, required=True)
    group.add_argument("--type", type=str, required=True, choices=["server", "client"])
    group.add_argument("--name", type=str, required=True)
    group.add_argument("--rank", type=int, required=True)
    group.add_argument("--size", type=int, required=True)

    return parser


def run_cli_command(cli_command: list, timeout: int = None):
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
        # env=custom_env
    )
    logger.info(f"return code: {cli_command_call.returncode}")

    if cli_command_call.returncode != 0:
        raise RuntimeError("Cli command returned code != 0")

    return cli_command_call.returncode


def api_command_wrapper(api_command_result, logger=None):
    """Prints the result of the command and raises RuntimeError to interrupt command sequence if there is an error.

    Args:
        api_command_result: result of the api command

    """
    if logger:
        logger.info("response: {}".format(api_command_result))
    if not api_command_result["status"] == "SUCCESS":
        if logger:
            logger.critical("command was not successful!")
        raise RuntimeError("command was not successful!")

    return api_command_result


def run_server(name, rank, size):
    """Runs the server communication process.

    Args:
        name (str): the name of the server
        rank (int): the rank of the server
        size (int): the size of the federation
        overseer (str, optional): the ip address of the overseer. Defaults to None.
    """
    logger = logging.getLogger()

    logger.info("****************** MCD INIT COMM ******************")

    # get the local ip of this current node
    local_hostname = socket.gethostname()
    local_ip = socket.gethostbyname(local_hostname)
    logger.info(f"Detected IP from socket.gethostbyname(): {local_ip}")

    # set self as overseer
    overseer_name = name
    overseer_ip = str(local_ip)

    # create hosts file to resolve ip adresses
    with (open("/etc/hosts", "a")) as f:
        # write server address
        f.write(f"{overseer_ip}\t{overseer_name}\n")

    # run server startup
    run_cli_command(["bash", "./startup/start.sh"])

    # we need to wait for startup to complete before calling admin port
    time.sleep(30)

    # communicate to clients through mlflow root (magic)
    mlflow_run = mlflow.start_run()
    mlflow_client = mlflow.tracking.client.MlflowClient()
    logger.info(f"run tags: {mlflow_run.data.tags}")
    logger.info(f"parent runId: {mlflow_run.data.tags.get('mlflow.parentRunId')}")
    root_run_id = mlflow_run.data.tags.get("mlflow.parentRunId")

    mlflow_client.set_tag(run_id=root_run_id, key="overseer_name", value=overseer_name)
    mlflow_client.set_tag(run_id=root_run_id, key="overseer_ip", value=overseer_ip)

    # let's start the...
    logger.info("****************** NVFLARE SUBMIT SEQUENCE ******************")

    logger.info("Starting FLAdminAPIRunner()")
    admin_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "admin@azure.ml"
    )
    os.makedirs(os.path.join(admin_dir, "local"), exist_ok=True)
    os.makedirs(os.path.join(admin_dir, "transfer"), exist_ok=True)

    run_cli_command(["nvflare", "preflight_check", "-p", admin_dir])

    runner = FLAdminAPIRunner(
        username="admin@azure.ml", admin_dir=admin_dir, debug=True
    )

    job_folder_name = os.path.join(admin_dir, "app")
    logger.info("Starting app from {}".format(job_folder_name))
    # see code from https://nvflare.readthedocs.io/en/2.2.1/_modules/nvflare/fuel/hci/client/fl_admin_api_runner.html
    try:
        # check status of the server
        def wait_for_client_connections(reply: FLAdminAPIResponse, **kwargs) -> bool:
            # wait for number of clients to be SIZE-1 (see args)
            if reply["details"][FLDetailKey.REGISTERED_CLIENTS] == (size - 1):
                return True
            else:
                return False

        logger.info("api.wait_until_server_status(TargetType.SERVER)")
        response = api_command_wrapper(
            runner.api.wait_until_server_status(
                timeout=600,  # let's give 10 mins for all clients to start
                interval=10,
                callback=wait_for_client_connections,
            ),
            logger,
        )
        logger.info("All clients are now connected to the server")

        # submit the job
        logger.info(f'api.submit_job("{job_folder_name}")')
        response = api_command_wrapper(runner.api.submit_job(job_folder_name), logger)
        job_id = response["details"]["job_id"]
        logger.info(f"NVFlare job_id={job_id}")

        # check server (again ?)
        time.sleep(30)
        logger.info("api.check_status(TargetType.SERVER)")
        api_command_wrapper(runner.api.check_status(TargetType.SERVER), logger)

        # wait until all clients are ready again (app is stopped).
        logger.info("api.wait_until_client_status()")
        api_command_wrapper(runner.api.wait_until_client_status(), logger)

        # check server and clients (again ?)
        logger.info("api.check_status(TargetType.SERVER)")
        api_command_wrapper(runner.api.check_status(TargetType.SERVER), logger)
        # now server engine status should be stopped
        time.sleep(
            10
        )  # wait for clients to stop in case they take longer than server to stop
        logger.info("api.check_status(TargetType.CLIENT)")
        api_command_wrapper(runner.api.check_status(TargetType.CLIENT), logger)

        # shutdown everything
        logger.info("api.shutdown(target_type=TargetType.ALL)")
        api_command_wrapper(runner.api.shutdown(target_type=TargetType.ALL), logger)

    except RuntimeError as e:
        err_msg = f"There was an exception during an admin api command: {secure_format_exception(e)}"
        raise RuntimeError(err_msg)


def run_client(name, rank, size):
    """Runs the client communication process.

    Args:
        name (str): the name of the server
        rank (int): the rank of the server
        size (int): the size of the federation
    """
    logger = logging.getLogger()

    # get Mlflow client and root run id
    mlflow_run = mlflow.start_run()
    mlflow_client = mlflow.tracking.client.MlflowClient()
    logger.debug(f"parent runId: {mlflow_run.data.tags.get('mlflow.parentRunId')}")
    root_run_id = mlflow_run.data.tags.get("mlflow.parentRunId")

    overseer_name = None
    overseer_ip = None
    fetch_start_time = time.time()

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

    # run client startup
    logger.info("Running ./startup/sub_start.sh")
    run_cli_command(["bash", "./startup/sub_start.sh"])


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

    logger.info("****************** MCD RUNTIME INIT ******************")
    logger.info("sys.argv: {}".format(sys.argv))

    # parse the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    logger.info("args: {}".format(args))

    # launch server / client
    if args.type == "server":
        logger.info("****************** MCD SERVER RUN ******************")
        run_server(args.name, args.rank, args.size)
    else:
        logger.info("****************** MCD CLIENT RUN ******************")
        run_client(args.name, args.rank, args.size)


if __name__ == "__main__":
    main()
