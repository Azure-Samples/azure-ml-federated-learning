import os
import sys
import logging
import argparse
import socket
import subprocess


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
    group.add_argument("--run_id", type=str, required=False)
    group.add_argument("--name", type=str, required=False, default=None)
    group.add_argument("--rank", type=int, required=True)
    group.add_argument("--size", type=int, required=True)

    return parser


def run_cli_command(cli_command: list, timeout: int = None, env: dict = None):
    """Runs subprocess for a cli setup command"""
    logger = logging.getLogger()
    logger.info(f"Launching cli with command: {cli_command}")
    cli_command_call = subprocess.run(
        cli_command,
        # stdout=PIPE,
        # stderr=PIPE,
        universal_newlines=True,
        check=False,  # will not raise an exception if subprocess fails
        timeout=timeout,
        env=env,
    )
    logger.info(f"return code: {cli_command_call.returncode}")

    if cli_command_call.returncode != 0:
        raise RuntimeError("CLI command returned code != 0")

    return cli_command_call.returncode


def run_head(sb_comm, name: str, rank: int, size: int) -> dict:
    """Runs the head communication process.

    Args:
        sb_comm (object): the service bus communicator instance
        name (str): the name of the head
        rank (int): the rank of the head
        size (int): the size of the federation

    Returns:
        dict: the federation config
    """
    logger = logging.getLogger()

    # get the local ip of this current node
    local_hostname = socket.gethostname()
    local_ip = socket.gethostbyname(local_hostname)
    logger.info(f"Detected IP from socket.gethostbyname(): {local_ip}")

    # create a "federation" config
    federation_config = {"head": {"name": name, "ip": local_ip}, "workers": {}}

    # create a config capturing all the workers ip adresses and names
    logger.info("Waiting for worker to connect...")
    for _rank in range(1, size):
        # ask each worker in sequence for its config
        logger.info("Waiting for worker {}...".format(_rank))
        worker_config = sb_comm.recv(source=_rank, tag="IP")
        logger.info("Received worker {} config: {}".format(_rank, worker_config))
        federation_config["workers"][worker_config["name"]] = worker_config["ip"]

    logger.info("Gathered federation config: {}".format(federation_config))

    # send this config to every worker
    for _rank in range(1, size):
        logger.info("Sending federation config to worker {}...".format(_rank))
        sb_comm.send(
            federation_config,
            target=_rank,
            tag="CONFIG",
        )

    # create hosts file to resolve ip adresses
    with (open("/etc/hosts", "a")) as f:
        # write head address
        f.write(str(local_ip) + "\t" + name + "\n")

        # write each worker adresses
        for worker_name, worker_ip in federation_config["workers"].items():
            f.write(str(worker_ip) + "\t" + worker_name + "\n")

    # we can run a setup command here?

    # now that head is ready, send an order for each worker to start
    for _rank in range(1, size):
        logger.info("Sending order to start to worker {}...".format(_rank))
        sb_comm.send("START", target=_rank, tag="START")

    return federation_config


def run_worker(sb_comm, name, rank, size) -> dict:
    """Runs the worker communication process.

    Args:
        sb_comm (object): the service bus communicator instance
        name (str): the name of the worker
        rank (int): the rank of the worker
        size (int): the size of the federation

    Returns:
        dict: the federation config
    """
    logger = logging.getLogger()

    # get the local ip of this current node
    local_hostname = socket.gethostname()
    local_ip = socket.gethostbyname(local_hostname)
    logger.info(f"Detected IP from socket.gethostbyname(): {local_ip}")

    # send the ip to the head (rank=0) and wait for the federation config in return
    sb_comm.send({"name": name, "ip": local_ip}, target=0, tag="IP")
    federation_config = sb_comm.recv(source=0, tag="CONFIG")
    logger.info("Received federation config: {}".format(federation_config))

    # create hosts file to resolve ip adresses
    with (open("/etc/hosts", "a")) as f:
        # write head address
        f.write(
            str(federation_config["head"]["ip"])
            + "\t"
            + federation_config["head"]["name"]
            + "\n"
        )

        # write each worker adresses
        for worker_name, worker_ip in federation_config["workers"].items():
            f.write(str(worker_ip) + "\t" + worker_name + "\n")

    # wait for start signal from rank=0
    sb_comm.recv(source=0, tag="START")

    # we can run a setup command here?

    return federation_config


def main():
    """Script main function."""
    # Create and configure logger to write into a file in job outputs/
    logging.basicConfig(
        filename="outputs/mcd_runtime.log",
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        filemode="a",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("****************** MCD RUNTIME INIT ******************")
    logger.info("sys.argv: {}".format(sys.argv))

    # parse the arguments
    parser = get_arg_parser()
    args, custom_command = parser.parse_known_args()
    logger.info("args: {}".format(args))

    if args.run_id is None:
        try:
            from azureml.core import Run

            args.run_id = Run.get_context().parent.id

        except:
            raise Exception(
                "Either manually specify parameter run_id or run the experiment online."
            )

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

        logger.info("****************** MCD INIT COMM ******************")
        sb_comm.initialize()

        if args.name is None:
            if args.rank == 0:
                args.name = "head"
            else:
                args.name = f"worker-{args.rank}"

        if args.rank == 0:
            # run the communication that needs to happen on a head node
            logger.info("****************** MCD SERVER RUN ******************")
            fed_config = run_head(sb_comm, args.name, args.rank, args.size)
        else:
            # run the communication that needs to happen on a worker node
            logger.info("****************** MCD CLIENT RUN ******************")
            fed_config = run_worker(sb_comm, args.name, args.rank, args.size)

    except BaseException as e:
        logger.critical("MCD RUNTIME ERROR: {}".format(e))
        raise e

    # run whatever goes after the known args as a command
    custom_env = dict(os.environ)  # create copy of env vars
    custom_env["MCD_RANK"] = str(args.rank)
    custom_env["MCD_SIZE"] = str(args.size)
    custom_env["MCD_RUN_ID"] = str(args.run_id)
    custom_env["MCD_CONFIG"] = str(fed_config)
    custom_env["MCD_HEAD"] = str(fed_config["head"]["ip"])
    custom_env["MCD_WORKERS"] = ",".join(
        [str(ip) for ip in fed_config["workers"].values()]
    )

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), "w") as env_file:
        env_file.write(
            f"DIST_GROUP_SIZE={args.size}\n"
            + f"DIST_GROUP_RANK={args.rank}\n"
            + f"DIST_GROUP_HOST_IP={custom_env['MCD_HEAD']}\n"
            + f"DIST_GROUP_CONTRIBUTORS_IP={custom_env['MCD_WORKERS']}"
        )
        env_file.close()

    if custom_command:
        run_cli_command(custom_command, env=custom_env)


if __name__ == "__main__":
    main()
