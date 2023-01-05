import os
import sys
import logging

# Create and configure logger
logging.basicConfig(
    filename="outputs/mcd_runtime.log",
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    filemode="a",
)
MCD_HOST_LOGGER = logging.getLogger()
MCD_HOST_LOGGER.setLevel(logging.DEBUG)

MCD_HOST_LOGGER.info("****************** MCD RUNTIME INIT ******************")
MCD_HOST_LOGGER.info("sys.argv: {}".format(sys.argv))

# UGLY, YOU CAN DO BETTER
MCD_RUN_ID = str(sys.argv[1])
MCD_RANK = int(sys.argv[2])
MCD_SIZE = int(sys.argv[3])

MCD_HOST_LOGGER.info("MCD_RANK={}".format(MCD_RANK))
MCD_HOST_LOGGER.info("MCD_SIZE={}".format(MCD_SIZE))
MCD_HOST_LOGGER.info("MCD_RUN_ID={}".format(MCD_RUN_ID))
MCD_COMMAND = sys.argv[4:]
MCD_HOST_LOGGER.info("MCD_COMMAND='{}'".format(MCD_COMMAND))

import socket

# network config
LOCAL_HOSTNAME = socket.gethostname()
LOCAL_IP = socket.gethostbyname(LOCAL_HOSTNAME)
MCD_HOST_LOGGER.info(f"Detected IP from socket.gethostbyname(): {LOCAL_IP}")

from service_bus_driver import ServiceBusMPILikeDriver

sb_comm = ServiceBusMPILikeDriver(
    world_size=MCD_SIZE,
    world_rank=MCD_RANK,
    topic="mcd",
    subscription=MCD_RUN_ID,
    auth_method="ConnectionString",
    allowed_tags=["IP", "CONFIG", "RUN", "KILL"],
)
try:
    sb_comm.initialize()

    if MCD_RANK == 0:
        head_ip = LOCAL_IP
        worker_ip_list = []
        MCD_HOST_LOGGER.info("Waiting for workers to connect...")
        for rank in range(1, MCD_SIZE):
            MCD_HOST_LOGGER.info("Waiting for worker {}...".format(rank))
            worker_config = sb_comm.recv(source=rank, tag="IP")
            MCD_HOST_LOGGER.info(
                "Received worker {} config: {}".format(rank, worker_config)
            )
            worker_ip_list.append(worker_config["worker_ip"])
        MCD_HOST_LOGGER.info("Received all workers config: {}".format(worker_ip_list))

        for rank in range(1, MCD_SIZE):
            MCD_HOST_LOGGER.info("Sending workers config to worker {}...".format(rank))
            sb_comm.send(
                {"head": LOCAL_IP, "workers": worker_ip_list}, target=rank, tag="CONFIG"
            )

        for rank in range(1, MCD_SIZE):
            MCD_HOST_LOGGER.info(
                "Sending workers order to start to worker {}...".format(rank)
            )
            sb_comm.send(
                {"head": LOCAL_IP, "workers": worker_ip_list}, target=rank, tag="CONFIG"
            )

    else:
        sb_comm.send(
            {"worker_ip": LOCAL_IP, "worker_rank": MCD_RANK}, target=0, tag="IP"
        )
        mcd_config = sb_comm.recv(source=0, tag="CONFIG")
        worker_ip_list = mcd_config["workers"]
        head_ip = mcd_config["head"]

except BaseException as e:
    MCD_HOST_LOGGER.critical("MCD RUNTIME ERROR: {}".format(e))
    raise e


MCD_HOST_LOGGER.info("****************** MCD RUNTIME RUN ******************")

import subprocess

try:
    mcd_env = dict(os.environ)
    mcd_env["MCD_RANK"] = str(MCD_RANK)
    mcd_env["MCD_SIZE"] = str(MCD_SIZE)
    mcd_env["MCD_RUN_ID"] = str(MCD_RUN_ID)
    mcd_env["MCD_WORKERS"] = ",".join(worker_ip_list)
    mcd_env["MCD_HEAD"] = str(head_ip)

    with(open("/etc/hosts", "a")) as f:
        f.write(str(head_ip)+"\thead\n")
        for index, ip in enumerate(worker_ip_list):
            f.write(str(ip)+f"\tworker-{index}\n")

    subprocess.check_call(" ".join(MCD_COMMAND), shell=True, env=mcd_env)
    # proc = subprocess.check_call(
    #     MCD_COMMAND,
    #     shell=True,
    #     env=mcd_env,
    #     # stdout=subprocess.PIPE,
    #     # stderr=subprocess.STDOUT
    # )
    # while proc.poll() is None:
    #     output = proc.stdout.readline()
    #     print(output)
    # for line in iter(p.stdout.readline, b''):
    #     print(">>> " + line.rstrip())

except subprocess.CalledProcessError as e:
    MCD_HOST_LOGGER.critical("MCD RUNTIME ERROR: {}".format(e))
    sys.exit(e.returncode)

MCD_HOST_LOGGER.info("****************** MCD RUNTIME TEARDOWN ******************")

try:
    if MCD_RANK == 0:
        head_ip = LOCAL_IP
        worker_ip_list = []
        MCD_HOST_LOGGER.info("Sending KILL signal to workers...")
        for rank in range(1, MCD_SIZE):
            sb_comm.send("KILL", target=rank, tag="KILL")

    else:
        MCD_HOST_LOGGER.info("Waiting for KILL signal from head...")
        sb_comm.recv(source=0, tag="KILL")

    sb_comm.finalize()

except BaseException as e:
    MCD_HOST_LOGGER.critical("MCD RUNTIME ERROR: {}".format(e))
    raise e
