import pickle
import sys
import time
import sys
import logging
import socket

import mlflow

from enum import Enum

# Set logging to sys.out
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(log_format)
logger.addHandler(handler)


class FLAGS(Enum):
    """Enumerates type of messages that can be exchanged between peers"""

    SIZE = 0
    SHAPE = 1
    DATA = 2
    CONN = 3
    OK_SIZE = 10
    OK_SHAPE = 11
    OK_DATA = 12
    OK_CONN = 13
    FAIL = 999


DEFAULT_MSG_SIZE = 4096


class AMLComm:
    """AMLComm provides simple communication layer across nodes in AzureML.
    The communication capabilities are limited to scatter-gather pattern,
    where node 0 is considered as a host(master) node. Communication
    channel is established over Python sockets and data are serialized
    using pickle library.
    """

    def __init__(self, rank, world_size, run_id, encryption=None) -> None:
        """Initializes AMLComm communicator

        Args:
            rank: rank of the current node, must be between 0 and world_size-1
            world_size: number of nodes in our setup, must be positive integer larger than 1
            run_id: specifier of the run share across the nodes
            encryption (Optional): encryption used for messaging (must expose API like AMLSPMC)
        """

        assert rank >= 0 and rank <= world_size - 1
        assert world_size > 1

        self._rank = rank
        self._world_size = world_size
        self._run_id = run_id
        self._connections = {}
        self._encryption = None
        self._setup_master()
        self._setup_encryption(encryption)

    def _get_open_port(self):
        from contextlib import closing

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return str(s.getsockname()[1])

    def _setup_master(self):
        host_ip, host_port = None, None

        with mlflow.start_run() as mlflow_run:
            mlflow_client = mlflow.tracking.client.MlflowClient()
            if self._rank == 0:
                host_port = self._get_open_port()
                local_hostname = socket.gethostname()
                host_ip = str(socket.gethostbyname(local_hostname))
                mlflow_client.set_tag(
                    run_id=self._run_id, key="aml_host_ip", value=host_ip
                )
                mlflow_client.set_tag(
                    run_id=self._run_id, key="aml_host_port", value=host_port
                )
            else:
                host_ip, host_port = None, None
                fetch_start_time = time.time()

                while host_ip is None or host_port is None:
                    logger.info(f"Checking out tag aml_host_ip and aml_host_port...")
                    mlflow_root_run = mlflow_client.get_run(self._run_id)

                    if "aml_host_ip" in mlflow_root_run.data.tags:
                        host_ip = mlflow_root_run.data.tags["aml_host_ip"]
                        logger.info(f"host_ip found: {host_ip}")

                    if "aml_host_port" in mlflow_root_run.data.tags:
                        host_port = mlflow_root_run.data.tags["aml_host_port"]
                        logger.info(f"host_port found: {host_port}")

                    if (host_ip is None) and (time.time() - fetch_start_time > 600):
                        raise RuntimeError("Could not fetch the tag within timeout.")
                    else:
                        time.sleep(10)

        self._host_ip = host_ip
        self._host_port = host_port
        self._local_ip = str(socket.gethostbyname(socket.gethostname()))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(1000)  # Set timeout so receive does nto wait forever

        if self._rank == 0:
            self._socket.bind(("", int(self._host_port)))
            for _ in range(self._world_size - 1):
                self._socket.listen(1)
                conn, addr = self._socket.accept()
                logger.info(f"Connected to: {addr}")

                while True:
                    try:
                        msg = conn.recv(1024)
                        if not msg:
                            break

                        msg = pickle.loads(msg)
                        if msg["flag"] != FLAGS.CONN:
                            break
                        logger.info(f"Received data: {msg}")
                        # Add client with its rank to our dictionary of connections
                        self._connections[int(msg["data"])] = (conn, addr)
                        break
                    except socket.error as e:
                        logger.exception("Socket error: {e}")
                        break
            logger.info(f"Number of connections: {len(self._connections)}")
            assert len(self._connections) == self._world_size - 1
            for c in self._connections:
                conn = self._connections[c][0]
                bytes_str = pickle.dumps({"flag": FLAGS.OK_CONN, "data": None})
                conn.sendall(bytes_str)
        else:
            self._socket.connect((self._host_ip, int(self._host_port)))
            bytes_str = pickle.dumps({"flag": FLAGS.CONN, "data": self._rank})
            self._socket.sendall(bytes_str)
            msg = pickle.loads(self._socket.recv(1024))
            logger.info(f"Received from host: {msg}")
            assert msg["flag"] == FLAGS.OK_CONN

    def _setup_encryption(self, encryption):
        if encryption is None:
            return

        pub_key = encryption.get_public_key()
        if self._rank == 0:
            for c in self._connections:
                self.send(pub_key, c)
                encryption.add_remote_public_key(c, self.recv(c))

        else:
            encryption.add_remote_public_key(0, self.recv(0))
            self.send(pub_key, 0)

        self._encryption = encryption

    def _send(self, msg, destination, flag, ok_flag):
        assert destination != self._rank
        if self._rank != 0:
            assert destination == 0

        if self._rank == 0:
            conn = self._connections[destination][0]
        else:
            conn = self._socket

        time_start = time.time()
        tries = 0
        while True:
            try:
                conn.sendall(msg)

                response = conn.recv(1024)
                if self._encryption is not None:
                    response = self._encryption.decrypt(response)
                response = pickle.loads(response)

                if "flag" in response and response["flag"] == ok_flag:
                    break
                else:
                    logger.exception(
                        f"Message does not match ok flag {ok_flag}, received message: {response}"
                    )
            except Exception as e:
                logger.exception(e)

                tries += 1
                if time.time() - time_start >= 60 or tries >= 3:
                    raise Exception(
                        f"Failed sending message to {destination}, flag: {flag}", e
                    )

    def _receive(self, source, flag, ok_flag, msg_size=None):
        assert source != self._rank
        if self._rank != 0:
            assert source == 0

        if self._rank == 0:
            conn = self._connections[source][0]
        else:
            conn = self._socket

        if msg_size is None:
            packet_max_size = DEFAULT_MSG_SIZE
        else:
            packet_max_size = msg_size

        time_start = time.time()
        data = None
        tries = 0
        while True:
            try:
                msg = b""
                # The socket may use buffers smaller than suggested size
                # and thus we may receive multiple of them
                while sys.getsizeof(msg) < packet_max_size:
                    packet = conn.recv(packet_max_size)
                    if not packet:
                        break
                    msg += packet
                    if msg_size is None:
                        break

                if self._encryption is not None:
                    msg = self._encryption.decrypt(msg)
                msg = pickle.loads(msg)
                if "flag" in msg and msg["flag"] == flag:
                    data = msg["data"]
                    break
                else:
                    raise Exception(
                        f"Failed to receive message from {source}, flag: {flag}, data received: {msg}"
                    )
            except Exception as e:
                logger.exception(e)
                # Purge the socket buffer
                conn.setblocking(0)
                conn.setblocking(1)

                msg_fail = pickle.dumps({"flag": FLAGS.FAIL, "data": None})
                if self._encryption is not None:
                    msg_fail = self._encryption.encrypt(msg_fail, source)
                # Send information about failure
                conn.sendall(msg_fail)

                tries += 1
                if time.time() - time_start >= 60 and tries >= 3:
                    raise Exception(
                        f"Failed receiving message from {source}, flag: {flag}, msg: {msg}",
                        e,
                    )
                time.sleep(1)

        # Send confirmation about received payload size information
        msg_success = pickle.dumps({"flag": ok_flag, "data": None})
        if self._encryption is not None:
            msg_success = self._encryption.encrypt(msg_success, source)
        conn.sendall(msg_success)
        return data

    def send(self, data, destination):
        """Sends tensor to the destination contributor node

        Args:
            data: data to be sent
            destination: rank of the receiver node
        """

        # Get size of the payload
        msg_payload = pickle.dumps({"flag": FLAGS.DATA, "data": data})
        if self._encryption is not None:
            msg_payload = self._encryption.encrypt(msg_payload, destination)
        size = sys.getsizeof(msg_payload)

        # Notify destination about size of the payload and wait for confirmation
        msg_size = pickle.dumps({"flag": FLAGS.SIZE, "data": size})
        if self._encryption is not None:
            msg_size = self._encryption.encrypt(msg_size, destination)
        self._send(msg_size, destination, FLAGS.SIZE, FLAGS.OK_SIZE)

        # Send the payload
        self._send(msg_payload, destination, FLAGS.DATA, FLAGS.OK_DATA)

    def recv(self, source):
        """Receive tensor from the source rank node

        Args:
            source: rank of the sender node
        """

        # Receive size information about size of the payload
        size = self._receive(source, FLAGS.SIZE, FLAGS.OK_SIZE)
        # Receive payload
        tensor_data = self._receive(source, FLAGS.DATA, FLAGS.OK_DATA, size)
        return tensor_data

    def close(self):
        """Close the communication channels gracefully"""
        if self._rank == 0:
            for c in self._connections:
                self._connections[c][0].close()

        self._socket.close()
