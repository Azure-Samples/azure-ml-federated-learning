import os
import sys
import time
import logging
import argparse
import socket
import subprocess
from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner
from nvflare.fuel.hci.client.fl_admin_api_spec import (
    APISyntaxError,
    FLAdminAPIResponse,
    FLAdminAPISpec,
    TargetType,
)
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
    group.add_argument("--command", type=str, required=False, default=None)
    group.add_argument("--overseer", type=str, required=False, default=None)

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


def run_server(sb_comm, name, rank, size, overseer=None):
    """Runs the server communication process.

    Args:
        sb_comm (object): the service bus communicator instance
        name (str): the name of the server
        rank (int): the rank of the server
        size (int): the size of the federation
        overseer (str, optional): the ip address of the overseer. Defaults to None.
    """
    logger = logging.getLogger()

    logger.info("****************** MCD INIT COMM ******************")
    sb_comm.initialize()

    # get the local ip of this current node
    local_hostname = socket.gethostname()
    local_ip = socket.gethostbyname(local_hostname)
    logger.info(f"Detected IP from socket.gethostbyname(): {local_ip}")

    # create a "federation" config
    federation_config = {"server": {"name": name, "ip": local_ip}, "clients": {}}

    # create a config capturing all the clients ip adresses and names
    logger.info("Waiting for client to connect...")
    for _rank in range(1, size):
        # ask each client in sequence for its config
        logger.info("Waiting for client {}...".format(_rank))
        client_config = sb_comm.recv(source=_rank, tag="IP")
        logger.info("Received client {} config: {}".format(_rank, client_config))
        federation_config["clients"][client_config["name"]] = client_config["ip"]

    logger.info("Gathered federation config: {}".format(federation_config))

    # send this config to every client
    for _rank in range(1, size):
        logger.info("Sending federation config to client {}...".format(_rank))
        sb_comm.send(
            federation_config,
            target=_rank,
            tag="CONFIG",
        )

    # create hosts file to resolve ip adresses
    with (open("/etc/hosts", "a")) as f:
        # write server address
        f.write(str(local_ip) + "\t" + name + "\n")

        # write each client adresses
        for client_name, client_ip in federation_config["clients"].items():
            f.write(str(client_ip) + "\t" + client_name + "\n")

        # write overseer
        if overseer:
            f.write(str(overseer) + "\t" + "overseer" + "\n")

    # run server startup
    logger.info("Running ./startup/start.sh")
    # TODO: run sub_start.sh instead and figure out logs pipe
    run_cli_command(["bash", "./startup/start.sh"])

    # now that server is ready, send an order for each client to start
    for _rank in range(1, size):
        logger.info("Sending order to start to client {}...".format(_rank))
        sb_comm.send("START", target=_rank, tag="START")

    # we don't need this beyond this point
    sb_comm.finalize()

    # let's start the...
    logger.info("****************** NVFLARE SUBMIT SEQUENCE ******************")

    # we need to wait for startup to complete before calling admin port
    time.sleep(30)

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


def run_client(sb_comm, name, rank, size, overseer=None):
    """Runs the client communication process.

    Args:
        sb_comm (object): the service bus communicator instance
        name (str): the name of the server
        rank (int): the rank of the server
        size (int): the size of the federation
        overseer (str, optional): the ip address of the overseer. Defaults to None.
    """
    logger = logging.getLogger()

    logger.info("****************** MCD INIT COMM ******************")
    sb_comm.initialize()

    # get the local ip of this current node
    local_hostname = socket.gethostname()
    local_ip = socket.gethostbyname(local_hostname)
    logger.info(f"Detected IP from socket.gethostbyname(): {local_ip}")

    # send the ip to the server (rank=0) and wait for the federation config in return
    sb_comm.send({"name": name, "ip": local_ip}, target=0, tag="IP")
    federation_config = sb_comm.recv(source=0, tag="CONFIG")
    logger.info("Received federation config: {}".format(federation_config))

    if "server" not in federation_config:
        raise ValueError("Federation config does not contain server information")
    if (
        "ip" not in federation_config["server"]
        or "name" not in federation_config["server"]
    ):
        raise ValueError("Federation config does not contain server information")

    # create hosts file to resolve ip adresses
    with (open("/etc/hosts", "a")) as f:
        # write server address
        f.write(
            str(federation_config["server"]["ip"])
            + "\t"
            + federation_config["server"]["name"]
            + "\n"
        )

        # write each client adresses
        for client_name, client_ip in federation_config["clients"].items():
            f.write(str(client_ip) + "\t" + client_name + "\n")

        # write overseer
        if overseer:
            f.write(str(overseer) + "\t" + "overseer" + "\n")

    # wait for start signal from rank=0
    sb_comm.recv(source=0, tag="START")

    # we don't need this beyond this point
    sb_comm.finalize()

    # run client startup
    logger.info("Running ./startup/sub_start.sh")
    run_cli_command(["bash", "./startup/sub_start.sh"])

    # send start signal back
    # sb_comm.send("START", target=0, tag="START")


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

    # use service bus to communicate with other nodes
    try:
        from service_bus_driver import ServiceBusMPILikeDriver

        sb_comm = ServiceBusMPILikeDriver(
            world_size=args.size,
            world_rank=args.rank,
            topic="mcd",
            subscription=args.run_id,
            auth_method="ConnectionString",
            allowed_tags=["IP", "CONFIG", "START", "KILL"],
        )

        if args.type == "server":
            # run the communication that needs to happen on a server node
            logger.info("****************** MCD SERVER RUN ******************")
            run_server(sb_comm, args.name, args.rank, args.size, overseer=args.overseer)
        else:
            # run the communication that needs to happen on a client node
            logger.info("****************** MCD CLIENT RUN ******************")
            run_client(sb_comm, args.name, args.rank, args.size, overseer=args.overseer)

    except BaseException as e:
        logger.critical("MCD RUNTIME ERROR: {}".format(e))
        raise e


if __name__ == "__main__":
    main()
