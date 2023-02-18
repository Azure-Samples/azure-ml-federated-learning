from abc import ABC
import sys
import time
import pickle
import logging
import socket

from enum import Enum

import mlflow

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


class AMLComm(ABC):
    def __init__(self, rank, world_size, run_id) -> None:
        """Initializes AMLComm communicator

        Args:
            rank: rank of the current node, must be between 0 and world_size-1
            world_size: number of nodes in our setup, must be positive integer larger than 1
            run_id: specifier of the run share across the nodes
        """
        super().__init__()

        assert rank >= 0 and rank <= world_size - 1
        assert world_size > 1

        self._rank = rank
        self._world_size = world_size
        self._run_id = run_id
        self._stats = {
            "msg_received": 0,
            "msg_sent": 0,
            "sending_time": 0.0,
            "receiving_time": 0.0,
            "waiting_time": 0.0,
        }

    def log_stats(self, mlflow_client: mlflow.MlflowClient):
        self._stats["sending_time_avg"] = self._stats["sending_time"] / float(
            self._stats["msg_sent"]
        )
        self._stats["receiving_time_avg_w_waiting"] = self._stats[
            "receiving_time"
        ] / float(self._stats["msg_received"])
        self._stats["receiving_time_avg_wo_waiting"] = (
            self._stats["receiving_time"] - self._stats["waiting_time"]
        ) / float(self._stats["msg_received"])

        for key, value in self._stats.items():
            mlflow_client.log_metric(
                self._run_id,
                f"{self.__class__.__name__}_rank_{self._rank}_{key}",
                value,
            )

        mlflow_client.log_dict(
            self._run_id,
            self._stats,
            f"{self.__class__.__name__}_rank_{self._rank}_stats_summary.json",
        )


class AMLCommSocket(AMLComm):
    """AMLComm provides simple communication layer across nodes in AzureML.
    The communication capabilities are limited to scatter-gather pattern,
    where node 0 is considered as a host(master) node. Communication
    channel is established over Python sockets and data are serialized
    using pickle library.
    """

    def __init__(self, rank, world_size, run_id, host_ip=None, host_port=None) -> None:
        """Initializes AMLComm communicator

        Args:
            rank: rank of the current node, must be between 0 and world_size-1
            world_size: number of nodes in our setup, must be positive integer larger than 1
            run_id: specifier of the run share across the nodes
            host_ip (Optional): IP address of the host node, if not provided MLFlow is used to communicate it
            host_port (Optional): port of the host node, if not provided MLFlow is used to communicate it
        """
        super(AMLCommSocket, self).__init__(rank, world_size, run_id)

        self._host_ip, self._host_port = host_ip, host_port
        self._connections = {}
        self._setup_master()

    def _get_open_port(self):
        from contextlib import closing

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return str(s.getsockname()[1])

    def _setup_master(self):
        if self._host_ip is None or self._host_port is None:
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
                        logger.info(
                            f"Checking out tag aml_host_ip and aml_host_port..."
                        )
                        mlflow_root_run = mlflow_client.get_run(self._run_id)

                        if "aml_host_ip" in mlflow_root_run.data.tags:
                            host_ip = mlflow_root_run.data.tags["aml_host_ip"]
                            logger.info(f"host_ip found: {host_ip}")

                        if "aml_host_port" in mlflow_root_run.data.tags:
                            host_port = mlflow_root_run.data.tags["aml_host_port"]
                            logger.info(f"host_port found: {host_port}")

                        if (host_ip is None) and (time.time() - fetch_start_time > 600):
                            raise RuntimeError(
                                "Could not fetch the tag within timeout."
                            )
                        else:
                            time.sleep(10)

            self._host_ip = host_ip
            self._host_port = host_port

        self._local_ip = str(socket.gethostbyname(socket.gethostname()))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(
            600.0
        )  # Set timeout to 30 minutes so we do not wait for any request indefinitely

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

    def _send(self, data, destination, flag, ok_flag):
        assert destination != self._rank
        if self._rank != 0:
            assert destination == 0

        if self._rank == 0:
            conn = self._connections[destination][0]
        else:
            conn = self._socket

        time_start = time.time()
        tries = 0
        while tries < 3:
            try:
                msg = pickle.dumps({"flag": flag, "data": data})
                conn.sendall(msg)

                msg = conn.recv(1024)
                msg = pickle.loads(msg)
                if "flag" in msg and msg["flag"] == ok_flag:
                    break
                else:
                    raise Exception(
                        f"Message does not match ok flag {ok_flag}, received message: {msg}"
                    )
            except Exception as e:
                logger.exception(e)
                tries += 1
                continue

        if type(msg) != dict or "flag" not in msg or msg["flag"] != ok_flag:
            raise Exception(
                f"Failed sending message to {destination}, flag: {flag}, data: {data}"
            )

        self._stats["msg_sent"] += 1
        self._stats["sending_time"] += time.time() - time_start

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
        while tries < 3:
            try:
                msg = b""
                # The socket may use buffers smaller than suggested size
                # and thus we may receive multiple of them
                while sys.getsizeof(msg) < packet_max_size:
                    packet = conn.recv(packet_max_size)
                    self._stats["waiting_time"] += time.time() - time_start
                    if not packet:
                        break
                    msg += packet
                    if msg_size is None:
                        break

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
                time.sleep(1)
                conn.setblocking(1)
                tries += 1
                # Send information about failure
                conn.sendall(pickle.dumps({"flag": FLAGS.FAIL, "data": None}))
                continue

        if data is None:
            raise Exception(
                f"Failed receiving message from {source}, flag: {flag}, msg: {msg}"
            )

        # Send confirmation about received payload size information
        bytes_str = pickle.dumps({"flag": ok_flag, "data": None})
        conn.sendall(bytes_str)

        self._stats["msg_received"] += 1
        self._stats["receiving_time"] += time.time() - time_start

        return data

    def send(self, data, destination):
        """Sends tensor to the destination contributor node

        Args:
            data: data to be sent
            destination: rank of the receiver node
        """

        # Get size of the payload
        bytes_str_tensor = pickle.dumps({"flag": FLAGS.DATA, "data": data})
        size = sys.getsizeof(bytes_str_tensor)

        # Notify destination about size of the payload and wait for confirmation
        self._send(size, destination, FLAGS.SIZE, FLAGS.OK_SIZE)

        # Send the payload
        self._send(data, destination, FLAGS.DATA, FLAGS.OK_DATA)

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
        logger.info("Closing AMLCommSocket clients")
        if self._rank == 0:
            for c in self._connections:
                self._connections[c][0].close()

        self._socket.close()

    def __del__(self):
        """Close the communication channels gracefully on object dereferencing"""
        self.close()
