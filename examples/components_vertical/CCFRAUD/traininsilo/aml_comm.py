import io
import pickle
import ast
import sys
import time
import sys
import logging
import socket

import mlflow
import torch

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
    SIZE = 0
    SHAPE = 1
    DATA = 2
    CONN = 3
    OK_SIZE = 10
    OK_SHAPE = 11
    OK_DATA = 12
    OK_CONN = 13


class AMLComm:
    def __init__(self, rank, world_size, run_id, is_local=False) -> None:
        self._rank = rank
        self._world_size = world_size
        self._run_id = run_id
        self._is_local = is_local
        self._connections = {}
        self._setup_master()

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

            if self._rank == 0:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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

        msg = pickle.dumps({"flag": flag, "data": data})
        time_start = time.time()
        while time.time() - time_start < 10:
            pass

    def send(self, tensor, destination):
        """Sends tensor to the destination contributor node

        Args:
            tensor: PyTorch tensor to be sent
            destination: id of the node that should receive the tensor
        """
        assert destination != self._rank
        if self._rank != 0:
            assert destination == 0

        if self._rank == 0:
            conn = self._connections[destination][0]
        else:
            conn = self._socket

        # Notify destination about size of the payload and wait for confirmation
        bytes_str_tensor = pickle.dumps({"flag": FLAGS.DATA, "data": tensor})
        bytes_str = pickle.dumps(
            {"flag": FLAGS.SIZE, "data": sys.getsizeof(bytes_str_tensor)}
        )
        conn.sendall(bytes_str)

        # Payload size confirmation
        msg = conn.recv(1024)
        msg = pickle.loads(msg)
        assert "flag" in msg and msg["flag"] == FLAGS.OK_SIZE

        # Send tensor payload
        conn.sendall(bytes_str_tensor)

        # Tensor data confirmation
        msg = conn.recv(1024)
        msg = pickle.loads(msg)
        assert "flag" in msg and msg["flag"] == FLAGS.OK_DATA

    def recv(self, source):
        assert source != self._rank
        if self._rank != 0:
            assert source == 0

        if self._rank == 0:
            conn = self._connections[source][0]
        else:
            conn = self._socket

        # Receive payload size information
        msg = conn.recv(1024)
        msg = pickle.loads(msg)
        assert "flag" in msg and msg["flag"] == FLAGS.SIZE
        size = msg["data"]

        # Send confirmation about received payload size information
        bytes_str = pickle.dumps({"flag": FLAGS.OK_SIZE, "data": None})
        conn.sendall(bytes_str)

        # Receive payload
        msg = b""
        while sys.getsizeof(msg) < size:
            packet = conn.recv(size)
            if not packet:
                break
            msg += packet
        msg = pickle.loads(msg)
        assert "flag" in msg and msg["flag"] == FLAGS.DATA
        tensor_data = msg["data"]

        # Confirm successful data transfer
        bytes_str = pickle.dumps({"flag": FLAGS.OK_DATA, "data": None})
        conn.sendall(bytes_str)

        return tensor_data

    def close(self):
        if self._rank == 0:
            for c in self._connections:
                self._connections[0].close()

        self._socket.close()
