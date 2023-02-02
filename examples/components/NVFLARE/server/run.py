"""This script runs an NVFlare server inside an AzureML job.

The script will:
- run the server workspace setup script,
- publish the local ip as a pipeline root tag using mlflow,
- wait for clients to connect,
- connect to the NVFlare server using admin api,
- submit the NVFlare app to itself.
"""
import os
import sys
import time
import logging
import argparse
import socket
import subprocess
import tempfile
import shutil
from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import (
    FLAdminAPIResponse,
    TargetType,
)
from nvflare.security.logging import secure_format_exception
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
    group.add_argument(
        "--federation_identifier",
        type=str,
        required=True,
        help="a unique identifier for the group of clients and server to find each other",
    )
    group.add_argument(
        "--server_config",
        type=str,
        required=True,
        help="the NVFlare workspace folder for this server",
    )
    group.add_argument(
        "--admin_config",
        type=str,
        required=True,
        help="the NVFlare workspace admin folder to connect to the server",
    )
    group.add_argument(
        "--app_dir", type=str, required=True, help="the NVFlare app code directory"
    )
    group.add_argument(
        "--server_name",
        type=str,
        required=True,
        help="the name of the server/overseer expected by clients for hostname resolution",
    )
    group.add_argument(
        "--expected_clients",
        type=int,
        required=True,
        help="the number of clients expected to connect to the server before training",
    )
    group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="where the NVFlare job artefacts will be saved upon completion of the job",
    )
    group.add_argument(
        "--wait_for_clients_timeout",
        type=int,
        required=False,
        default=600,
        help="the number of seconds to wait for clients to connect before timing out",
    )

    return parser


def run_cli_command(cli_command: list, timeout: int = None):
    """Runs subprocess for a cli setup command"""
    logger = logging.getLogger()
    logger.info(f"Launching cli with command: {cli_command}")
    cli_command_call = subprocess.run(
        cli_command,
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


def publish_server_ip(
    mlflow_run,
    federation_identifier,
    server_name=None,
    target_run_tag="mlflow.rootRunId",
):
    """Publishes the server ip to mlflow as a tag.

    Args:
        mlflow_run (mlflow.entities.Run): the mlflow run object
        federation_identifier (str): the unique identifier for the federation
        server_name (str, optional): the name of the server to publish for the clients
        target_run_tag (str, optional): the tag to use to find the root run. Defaults to "mlflow.rootRunId".

    Returns:
        str, str: the server ip and name
    """
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

    # publish the server ip/name as a tag in mlflow
    mlflow_client.set_tag(
        run_id=target_run_id, key=f"{federation_identifier}.server_ip", value=server_ip
    )

    if server_name:
        mlflow_client.set_tag(
            run_id=target_run_id,
            key=f"{federation_identifier}.server_name",
            value=server_name,
        )

    return server_ip, server_name


def run_server(
    federation_identifier,
    server_config_dir,
    admin_config_dir,
    app_dir,
    server_name,
    expected_clients,
    output_dir,
    wait_for_clients_timeout=600,
):
    """Runs the server communication process.

    Args:
        federation_identifier (str): the unique identifier for the federation
        server_config_dir (str): the path to the server config directory (from provision)
        admin_config_dir (str): the path to the admin config directory (from provision)
        app_dir (str): the path to the NVFlare application (training code) to submit
        server_name (str): the name of the server
        expected_clients (int): the size of the federation
        output_dir (str): the path to the output directory for the NVFlare job
        wait_for_clients_timeout (int, optional): the number of seconds to wait for clients to connect before timing out. Defaults to 600.
    """
    logger = logging.getLogger()

    # organize files locally
    workspace_dir = tempfile.TemporaryDirectory().name
    logger.info(f"workspace_dir: {workspace_dir}")

    # copy server config locally
    server_config_dir_local = os.path.join(workspace_dir, "server_config")
    shutil.copytree(server_config_dir, server_config_dir_local)

    # copy admin config locally
    admin_config_dir_local = os.path.join(
        workspace_dir, "admin_config", "admin@azure.ml"
    )
    shutil.copytree(admin_config_dir, admin_config_dir_local)
    os.makedirs(os.path.join(admin_config_dir_local, "local"), exist_ok=True)
    os.makedirs(os.path.join(admin_config_dir_local, "transfer"), exist_ok=True)

    # copy job folder inside admin folder for some reason
    job_folder_name = os.path.join(admin_config_dir_local, "app")
    shutil.copytree(app_dir, job_folder_name)

    # run server startup
    startup_script_path = os.path.join(server_config_dir_local, "startup", "start.sh")
    logger.info(f"Running {startup_script_path}")
    run_cli_command(["bash", startup_script_path])

    # we need to wait for startup to complete before calling admin port
    time.sleep(30)

    # communicate to clients through mlflow root (magic)
    mlflow_run = mlflow.start_run()
    overseer_name = server_name
    overseer_ip, _ = publish_server_ip(
        mlflow_run, federation_identifier, server_name=server_name
    )

    # create hosts file to resolve ip adresses
    with open("/etc/hosts", "a") as f:
        # write server address
        f.write(f"{overseer_ip}\t{overseer_name}\n")

    # let's start the...
    logger.info("****************** NVFLARE SUBMIT SEQUENCE ******************")

    logger.info("Starting FLAdminAPIRunner()")

    run_cli_command(["nvflare", "preflight_check", "-p", admin_config_dir_local])

    runner = FLAdminAPIRunner(
        username="admin@azure.ml", admin_dir=admin_config_dir_local, debug=True
    )
    logger.info("Starting app from {}".format(job_folder_name))
    # see code from https://nvflare.readthedocs.io/en/2.2.1/_modules/nvflare/fuel/hci/client/fl_admin_api_runner.html
    try:
        # check status of the server
        def wait_for_client_connections(reply: FLAdminAPIResponse, **kwargs) -> bool:
            # wait for number of clients to be SIZE-1 (see args)
            if reply["details"][FLDetailKey.REGISTERED_CLIENTS] == expected_clients:
                return True
            else:
                return False

        logger.info("api.wait_until_server_status(TargetType.SERVER)")
        response = api_command_wrapper(
            runner.api.wait_until_server_status(
                timeout=wait_for_clients_timeout,
                interval=10,
                callback=wait_for_client_connections,
            ),
            logger,
        )
        if response.get("details", {}).get("message", None) == "Waited until timeout.":
            raise RuntimeError("Waited for clients until timeout.")
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

        # TODO: check if job was successful or not
        logging.info("api.show_errors(TargetType.CLIENT)")
        client_errors = api_command_wrapper(
            runner.api.show_errors(job_id, target_type=TargetType.CLIENT), logger
        )
        if (
            isinstance(client_errors["details"]["message"], str)
            and client_errors["details"]["message"] == "No errors."
        ):
            logging.info("No errors found on clients")
        else:
            report = "\n".join(
                [
                    entry["data"]
                    for entry in client_errors["raw"]["data"]
                    if entry["type"] == "string"
                ]
            )
            raise Exception(
                f"Errors were found on clients, check client logs to debug:\n{report}"
            )

        logging.info("api.show_errors(TargetType.CLIENT)")
        server_errors = api_command_wrapper(
            runner.api.show_errors(job_id, target_type=TargetType.SERVER), logger
        )
        if (
            isinstance(server_errors["details"]["message"], str)
            and server_errors["details"]["message"] == "No errors."
        ):
            logging.info("No errors found on server")
        else:
            report = "\n".join(
                [
                    entry["data"]
                    for entry in server_errors["raw"]["data"]
                    if entry["type"] == "string"
                ]
            )
            raise Exception(
                f"Errors were found on server, check server logs to debug:\n{report}"
            )

        # check server and clients (again ?)
        logger.info("api.check_status(TargetType.SERVER)")
        api_command_wrapper(runner.api.check_status(TargetType.SERVER), logger)
        # now server engine status should be stopped
        time.sleep(
            10
        )  # wait for clients to stop in case they take longer than server to stop
        logger.info("api.check_status(TargetType.CLIENT)")
        api_command_wrapper(runner.api.check_status(TargetType.CLIENT), logger)

        # shutdown clients
        logger.info("api.shutdown(target_type=TargetType.CLIENT)")
        api_command_wrapper(runner.api.shutdown(target_type=TargetType.CLIENT), logger)

        # download results
        response = api_command_wrapper(runner.api.download_job(job_id))
        logger.info("Copying job artefacts to output_dir")
        shutil.copytree(
            os.path.join(admin_config_dir_local, "transfer"),
            os.path.join(output_dir, "artefacts"),
        )

        # shutdown system
        logger.info("api.shutdown(target_type=TargetType.ALL)")
        api_command_wrapper(runner.api.shutdown(target_type=TargetType.ALL), logger)

    except RuntimeError as e:
        err_msg = f"There was an exception during an admin api command: {secure_format_exception(e)}"
        raise RuntimeError(err_msg)


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

    run_server(
        args.federation_identifier,
        args.server_config,
        args.admin_config,
        args.app_dir,
        args.server_name,
        args.expected_clients,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
