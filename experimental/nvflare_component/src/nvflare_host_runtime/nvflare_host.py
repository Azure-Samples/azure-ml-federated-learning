import os
import sys
import logging
import argparse
import socket
import subprocess
from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner, api_command_wrapper
from nvflare.fuel.hci.client.fl_admin_api_spec import APISyntaxError, FLAdminAPIResponse, FLAdminAPISpec, TargetType

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
    group.add_argument("--type", type=str, required=True, choices=["server","client"])
    group.add_argument("--name", type=str, required=True)
    group.add_argument("--rank", type=int, required=True)
    group.add_argument("--size", type=int, required=True)
    group.add_argument("--command", type=str, required=False)

    return parser

def run_cli_command(cli_command: list, timeout: int = 60):
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

def run_server(sb_comm, name, rank, size):
    logger = logging.getLogger()

    LOCAL_HOSTNAME = socket.gethostname()
    LOCAL_IP = socket.gethostbyname(LOCAL_HOSTNAME)
    logger.info(f"Detected IP from socket.gethostbyname(): {LOCAL_IP}")

    clients_map = {}
    logger.info("Waiting for client to connect...")
    for _rank in range(1, size):
        logger.info("Waiting for client {}...".format(_rank))
        client_config = sb_comm.recv(source=_rank, tag="IP")
        logger.info(
            "Received client {} config: {}".format(_rank, client_config)
        )
        clients_map[client_config["name"]] = client_config["ip"]

    logger.info("Received all client config: {}".format(clients_map))

    for _rank in range(1, size):
        logger.info("Sending federation config to client {}...".format(_rank))
        sb_comm.send(
            {"server": {"name": name, "ip": LOCAL_IP}, "clients": clients_map}, target=_rank, tag="CONFIG"
        )

    with(open("/etc/hosts", "a")) as f:
        f.write(str(LOCAL_IP)+"\t"+name+"\n")
        for client_name, client_ip in clients_map.items():
            f.write(str(client_ip)+"\t"+client_name+"\n")

    # run server startup
    logger.info("Running ./startup/start.sh")
    run_cli_command(["bash", "./startup/start.sh"])

    for _rank in range(1, size):
        logger.info(
            "Sending order to start to client {}...".format(_rank)
        )
        sb_comm.send("START", target=_rank, tag="START")

    logger.info("Starting FLAdminAPIRunner()")
    admin_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"admin@nvidia.com/")
    os.makedirs(os.path.join(admin_dir, "local"), exist_ok=True)
    os.makedirs(os.path.join(admin_dir, "transfer"), exist_ok=True)
    runner = FLAdminAPIRunner(
        username="admin@nvidia.com",
        admin_dir=admin_dir,
        debug=True
    )

    runner.run(os.path.join(os.path.abspath(os.path.dirname(__file__)),"app/"))
    # for _rank in range(1, size):
    #     logger.info(
    #         "Getting start confirmation from client {}...".format(_rank)
    #     )
    #     sb_comm.recv(source=_rank, tag="START")
    # logger.info("Shutdown ???")
    # api_command_wrapper(runner.api.shutdown(target_type=TargetType.CLIENT))


def run_client(sb_comm, name, rank, size):
    logger = logging.getLogger()

    LOCAL_HOSTNAME = socket.gethostname()
    LOCAL_IP = socket.gethostbyname(LOCAL_HOSTNAME)
    logger.info(f"Detected IP from socket.gethostbyname(): {LOCAL_IP}")

    sb_comm.send(
        {"name": name, "ip": LOCAL_IP}, target=0, tag="IP"
    )
    federation_config = sb_comm.recv(source=0, tag="CONFIG")
    logger.info("Received federation config: {}".format(federation_config))

    with(open("/etc/hosts", "a")) as f:
        f.write(str(federation_config["server"]["ip"])+"\t"+federation_config["server"]["name"]+"\n")
        for client_name, client_ip in federation_config["clients"].items():
            f.write(str(client_ip)+"\t"+client_name+"\n")

    # wait for start signal
    sb_comm.recv(source=0, tag="START")

    # run client startup
    logger.info("Running ./startup/sub_start.sh")
    run_cli_command(["bash", "./startup/sub_start.sh"], timeout=None)

    # send start signal back
    # sb_comm.send("START", target=0, tag="START")



def main():
    # Create and configure logger
    logging.basicConfig(
        filename="outputs/mcd_runtime.log",
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        filemode="a",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("****************** MCD RUNTIME INIT ******************")
    logger.info("sys.argv: {}".format(sys.argv))

    parser = get_arg_parser()
    args = parser.parse_args()

    logger.info("args: {}".format(args))

    from service_bus_driver import ServiceBusMPILikeDriver

    sb_comm = ServiceBusMPILikeDriver(
        world_size=args.size,
        world_rank=args.rank,
        topic="mcd",
        subscription=args.run_id,
        auth_method="ConnectionString",
        allowed_tags=["IP", "CONFIG", "START", "KILL"],
    )

    try:
        sb_comm.initialize()

        if args.type == "server":
            logger.info("****************** MCD SERVER RUN ******************")
            run_server(sb_comm, args.name, args.rank, args.size)
        else:
            logger.info("****************** MCD CLIENT RUN ******************")
            run_client(sb_comm, args.name, args.rank, args.size)

    except BaseException as e:
        logger.critical("MCD RUNTIME ERROR: {}".format(e))
        raise e

if __name__ == "__main__":
    main()
