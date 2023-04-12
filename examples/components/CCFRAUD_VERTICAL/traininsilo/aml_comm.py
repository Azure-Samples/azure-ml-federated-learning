##########################################################################################
#                                         WARNING                                        #
##########################################################################################
# Should this file change please update all copies of aml_comm.py file in the repository #
##########################################################################################

from abc import ABC
import os
import sys
import math
import time
import pickle
import logging
import socket
import redis

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
    def __init__(self, rank, world_size, run_id, encryption=None) -> None:
        """Initializes AMLComm communicator

        Args:
            rank: rank of the current node, must be between 0 and world_size-1
            world_size: number of nodes in our setup, must be positive integer larger than 1
            run_id: specifier of the run share across the nodes
            encryption (Optional): encryption used for messaging (must expose API like AMLSPMC)
        """
        super().__init__()

        assert rank >= 0 and rank <= world_size - 1
        assert world_size > 1

        self._rank = rank
        self._world_size = world_size
        self._run_id = run_id
        self._stats = {
            "send_cnt": 0,
            "recv_cnt": 0,
            "send_retries_cnt": 0,
            "recv_retries_cnt": 0,
            "send_time": 0.0,
            "recv_time": 0.0,
            "send_wait_time": 0.0,
            "recv_wait_time": 0.0,
        }

        self._encryption = None
        self._temp_encryption = encryption

    def after_connection(self):
        self._setup_encryption()

    def _setup_encryption(self):
        if self._temp_encryption is None:
            return

        pub_key = self._temp_encryption.get_public_key()
        if self._rank == 0:
            for c in range(1, self._world_size):
                self.send(pub_key, c)
                self._temp_encryption.add_remote_public_key(c, self.recv(c))

        else:
            self._temp_encryption.add_remote_public_key(0, self.recv(0))
            self.send(pub_key, 0)
        self._encryption = self._temp_encryption

    def log_stats(self, mlflow_client: mlflow.MlflowClient):  # pragma: no cover
        self._stats["send_time_avg_w_waiting"] = self._stats["send_time"] / float(
            self._stats["send_cnt"]
        )
        self._stats["send_time_avg_wo_waiting"] = (
            self._stats["send_time"] - self._stats["send_wait_time"]
        ) / float(self._stats["send_cnt"])
        self._stats["recv_time_avg_w_waiting"] = self._stats["recv_time"] / float(
            self._stats["recv_cnt"]
        )
        self._stats["recv_time_avg_wo_waiting"] = (
            self._stats["recv_time"] - self._stats["recv_wait_time"]
        ) / float(self._stats["recv_cnt"])

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

    def __init__(
        self,
        rank,
        world_size,
        run_id,
        host_ip=None,
        host_port=None,
        encryption=None,
        timeout=600,
    ) -> None:
        """Initializes AMLComm communicator

        Args:
            rank: rank of the current node, must be between 0 and world_size-1
            world_size: number of nodes in our setup, must be positive integer larger than 1
            run_id: specifier of the run share across the nodes
            host_ip (Optional): IP address of the host node, if not provided MLFlow is used to communicate it
            host_port (Optional): port of the host node, if not provided MLFlow is used to communicate it
            encryption (Optional): encryption used for messaging (must expose API like AMLSPMC)
            timeout (Optional): timeout for socket operations. Defaults to 600 seconds (10 minutes)
        """
        super(AMLCommSocket, self).__init__(
            rank, world_size, run_id, encryption=encryption
        )

        self._host_ip, self._host_port = host_ip, host_port
        self._timeout = timeout
        self._connections = {}
        self._setup_master()
        self.after_connection()

    def _get_open_port(self):  # pragma: no cover
        from contextlib import closing

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return str(s.getsockname()[1])

    def _setup_master(self):
        if self._host_ip is None or self._host_port is None:  # pragma: no cover
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
        self._socket.settimeout(self._timeout)

        if self._rank == 0:
            self._socket.bind(("", int(self._host_port)))
            for _ in range(self._world_size - 1):
                self._socket.listen(1)
                conn, addr = self._socket.accept()
                logger.info(f"Connected to: {addr}")

                while True:
                    try:
                        msg = conn.recv(1024)
                        if not msg:  # pragma: no cover
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
        if self._rank == 0:
            conn = self._connections[destination][0]
        else:
            conn = self._socket

        retries = 0
        while retries < 3:
            try:
                conn.sendall(data)

                msg = conn.recv(1024)
                if self._encryption is not None:
                    msg = self._encryption.decrypt(msg)
                msg = pickle.loads(msg)
                if "flag" in msg and msg["flag"] == ok_flag:
                    break
                else:
                    raise Exception(
                        f"Message does not match ok flag {ok_flag}, received message: {msg}"
                    )
            except Exception as e:
                logger.exception(e)
                retries += 1
                self._stats["send_retries_cnt"] += 1
                continue

        if type(msg) != dict or "flag" not in msg or msg["flag"] != ok_flag:
            raise Exception(
                f"Failed sending message to {destination}, flag: {flag}, data: {data}"
            )

    def _recv(self, source, flag, ok_flag, msg_size=None):
        if self._rank == 0:
            conn = self._connections[source][0]
        else:
            conn = self._socket

        if msg_size is None:
            packet_max_size = DEFAULT_MSG_SIZE
        else:
            packet_max_size = msg_size

        data = None
        retries = 0

        while retries < 3:
            try:
                msg = b""
                # The socket may use buffers smaller than suggested size
                # and thus we may receive multiple of them
                while sys.getsizeof(msg) < packet_max_size:
                    packet = conn.recv(packet_max_size)
                    if not packet:  # pragma: no cover
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
                time.sleep(1)
                conn.setblocking(1)
                retries += 1
                self._stats["recv_retries_cnt"] += 1
                # Send information about failure
                msg = pickle.dumps({"flag": FLAGS.FAIL, "data": None})
                if self._encryption is not None:
                    msg = self._encryption.encrypt(msg, source)
                conn.sendall(msg)
                continue

        if data is None:
            raise Exception(
                f"Failed receiving message from {source}, flag: {flag}, msg: {msg}"
            )

        # Send confirmation about received payload size information
        msg = pickle.dumps({"flag": ok_flag, "data": None})
        if self._encryption is not None:
            msg = self._encryption.encrypt(msg, source)
        conn.sendall(msg)

        return data

    def send(self, data, destination):
        """Sends data to the destination node

        Args:
            data: data to be sent
            destination: rank of the receiver node
        """
        assert destination != self._rank, "Cannot send data to self"
        if self._rank != 0:
            assert destination == 0, "Only rank 0 can send data to other nodes"

        time_start = time.time()

        # Get size of the payload
        msg_payload = pickle.dumps({"flag": FLAGS.DATA, "data": data})
        if self._encryption is not None:
            msg_payload = self._encryption.encrypt(msg_payload, destination)

        msg_size = pickle.dumps(
            {"flag": FLAGS.SIZE, "data": sys.getsizeof(msg_payload)}
        )
        if self._encryption is not None:
            msg_size = self._encryption.encrypt(msg_size, destination)
        self._stats["send_wait_time"] += time.time() - time_start

        # Notify destination about size of the payload and wait for confirmation
        self._send(msg_size, destination, FLAGS.SIZE, FLAGS.OK_SIZE)

        # Send the payload
        self._send(msg_payload, destination, FLAGS.DATA, FLAGS.OK_DATA)
        self._stats["send_cnt"] += 1
        self._stats["send_time"] += time.time() - time_start

    def recv(self, source):
        """Receives data from the source rank node

        Args:
            source: rank of the sender node
        """
        assert source != self._rank
        if self._rank != 0:
            assert source == 0

        time_start = time.time()

        # Receive size information about size of the payload
        size = self._recv(source, FLAGS.SIZE, FLAGS.OK_SIZE)
        self._stats["recv_wait_time"] += time.time() - time_start

        # Receive payload
        tensor_data = self._recv(source, FLAGS.DATA, FLAGS.OK_DATA, size)
        self._stats["recv_cnt"] += 1
        self._stats["recv_time"] += time.time() - time_start

        return tensor_data

    def _close(self):  # pragma: no cover
        if self._socket is None:
            return
        logger.info("Closing AMLCommSocket clients")
        if self._rank == 0:
            for c in self._connections:
                self._connections[c][0].close()

        self._socket.close()
        self._socket = None

    def __del__(self):
        """Close the communication channels gracefully on object dereferencing"""
        self._close()


class AMLCommRedis(AMLComm):
    def __init__(
        self,
        rank,
        world_size,
        run_id,
        connection_string=None,
        message_timeout=600,
        connect_timeout=1800,
        encryption=None,
    ) -> None:
        """Initializes the AMLCommRedis class

        The authentication to the Redis server is attempted in the following order:
        1. Using the passed in connection_string if not empty or None.
        2. Using keyvault secret amlcomm-redis-connection-string.
        3. Using environment variable AML_COMM_REDIS_CONNECTION_STRING.

        Args:
            rank (int): rank of the current node
            world_size (int): total number of nodes
            run_id (str): run id of the current run
            connection_string (str, optional): connection string to the Redis server. Defaults to None.
            message_timeout (int, optional): timeout for the Redis messages. Defaults to 60.
            connect_timeout (int, optional): timeout for the initial connection to other nodes. Defaults to 1800.
            encryption (Optional): encryption used for messaging (must expose API like AMLSPMC)
        """
        super().__init__(rank, world_size, run_id, encryption=encryption)

        if not connection_string:
            connection_string = self._get_connection_string()
        connection_string = self._format_connection_string(connection_string)
        self._client: redis.Redis = self._connect_to_redis(connection_string)
        self._timeout = message_timeout
        self._max_msg_size = 100 * 1024 * 1024  # 100 MB
        self._wait_for_connection(connect_timeout)
        self.after_connection()

    def _connect_to_redis(self, connection_string):
        return redis.Redis.from_url(
            connection_string,
            health_check_interval=60,
            socket_connect_timeout=60,
            retry_on_timeout=True,
            socket_keepalive=True,
        )

    def _wait_for_connection(self, connect_timeout: int):  # pragma: no cover
        if self._rank == 0:
            connected = []
            for i in range(1, self._world_size):
                time_start = time.time()
                while time.time() - time_start < connect_timeout:
                    session_id = self._get_session_id(i, self._rank)
                    message = self._client.xread(
                        {session_id: 0}, count=1, block=self._timeout * 1000
                    )
                    if len(message) > 0:
                        message_id, message = message[0][1][0]
                        if b"rank" in message and int(message[b"rank"]) == i:
                            self._client.xdel(session_id, message_id)
                            connected.append(i)
                            logger.info(f"Connected to {i}")
                            break

            if len(connected) != self._world_size - 1:
                raise Exception(
                    f"Some clients did not connect, connected clients: {connected}"
                )
            else:
                for i in range(1, self._world_size):
                    self._client.xadd(
                        self._get_session_id(self._rank, i), {"rank": "0"}
                    )
                logger.info("All clients connected")

        else:
            # Notify client 0 about connection
            session_id = self._get_session_id(self._rank, 0)
            self._client.xadd(session_id, {"rank": f"{self._rank}"})

            # Wait for confirmation from client 0
            time_start = time.time()
            session_id = self._get_session_id(0, self._rank)
            while time.time() - time_start < connect_timeout:
                message = self._client.xread(
                    {session_id: 0}, count=1, block=self._timeout * 1000
                )
                if len(message) > 0:
                    message_id, message = message[0][1][0]
                    if b"rank" in message and int(message[b"rank"]) == 0:
                        self._client.xdel(session_id, message_id)
                        logger.info(f"Connected to 0")
                        return

            raise Exception(f"Failed to connect to client 0")

    def _get_connection_string(self) -> str:  # pragma: no cover
        try:
            from azureml.core import Run, Keyvault

            logger.warning("Getting Redis connection string from Azure ML Key Vault")
            run = Run.get_context()
            ws = run.experiment.workspace
            kv: Keyvault = ws.get_default_keyvault()
            connection_string = kv.get_secret("amlcomm-redis-connection-string")
            logger.info("Got Redis connection string from Azure ML Key Vault")
            return connection_string
        except Exception as e:
            logger.warning("Failed to get connection string from Azure ML Key Vault")
            logger.warning(f"Exception: {e}")

        logger.info("Getting Redis connection string from environment variable")
        if "AML_COMM_REDIS_CONNECTION_STRING" in os.environ:
            logger.info("Got Redis connection string from environment variable")
            return os.environ["AML_COMM_REDIS_CONNECTION_STRING"]
        else:
            logger.warning("Failed to get connection string from environment variable")

        raise Exception("Failed to get Redis connection string")

    def _format_connection_string(self, connection_string: str) -> str:
        from urllib.parse import urlparse, parse_qs

        connection_string = (
            connection_string[: connection_string.index(",")]
            + "?"
            + connection_string[connection_string.index(",") + 1 :]
        )
        connection_string = connection_string.replace(",", "&")

        if parse_qs(urlparse(connection_string).query)["ssl"][0].lower() == "true":
            connection_string = "rediss://" + connection_string
            connection_string = connection_string.replace("&ssl=True", "")
        else:
            connection_string = "redis://" + connection_string
            connection_string = connection_string.replace("&ssl=False", "")
        connection_string = connection_string.replace("&abortConnect=True", "")
        connection_string = connection_string.replace("&abortConnect=False", "")
        return connection_string

    def _get_session_id(self, source: int, destination: int, flag: FLAGS = None):
        if flag is None:
            return f"{self._run_id}:{source}=>{destination}"
        return f"{self._run_id}:{source}=>{destination}:{FLAGS[flag].name}"

    def _send(self, data, session_id, destination) -> None:  # pragma: no cover
        """Sends data to the destination node

        Args:
            data: data to be sent
            session_id: session id
            destination: rank of the receiver node
        """
        time_start = time.time()
        retries = 0

        while time.time() - time_start < self._timeout and retries < 3:
            try:
                self._client.xadd(session_id, {"data": data})
                return
            except Exception as e:
                logger.exception(
                    Exception(f"There was problem delivering message to {destination}")
                )
                logger.exception(e)

                retries += 1
                self._stats["send_retries_cnt"] += 1
                if retries >= 3:
                    raise e

        e = Exception(f"Sending message to {destination} timed out")
        logger.exception(e)
        raise e

    def send(self, data, destination) -> None:
        """Sends data to the destination node

        Args:
            data: data to be sent
            destination: rank of the receiver node
        """
        assert destination != self._rank, "Cannot send data to self"
        if self._rank != 0:
            assert destination == 0, "Only rank 0 can send data to other nodes"

        time_start = time.time()

        session_id = self._get_session_id(self._rank, destination)
        binary_data = pickle.dumps(data)
        if self._encryption:
            binary_data = self._encryption.encrypt(binary_data, destination)

        packets_count = math.ceil(sys.getsizeof(binary_data) / self._max_msg_size)
        if self._encryption:
            packets_count_msg = packets_count.to_bytes(2, "big")
            packets_count_msg = self._encryption.encrypt(packets_count_msg, destination)
        else:
            packets_count_msg = packets_count

        self._stats["send_wait_time"] += time.time() - time_start
        self._send(packets_count_msg, session_id, destination)
        for i in range(packets_count):
            self._send(
                binary_data[i * self._max_msg_size : (i + 1) * self._max_msg_size],
                session_id,
                destination,
            )
        self._stats["send_cnt"] += 1
        self._stats["send_time"] += time.time() - time_start

    def _recv(self, session_id, source):  # pragma: no cover
        """Receives data from the source rank node

        Args:
            session_id: session id
            source: rank of the sender node
        """
        time_start = time.time()
        retries = 0

        while time.time() - time_start < self._timeout and retries < 3:
            try:
                message = self._client.xread(
                    {session_id: 0}, count=1, block=self._timeout * 1000
                )
                if len(message) > 0:
                    # Get first message received [0]
                    # get contents of that message [1]
                    # get first message in that list [0]
                    message_id, message = message[0][1][0]
                    self._client.xdel(session_id, message_id)
                    return message[b"data"]
            except Exception as e:
                logger.exception(
                    Exception(f"There was problem receiving message from {source}")
                )
                logger.exception(e)

                retries += 1
                self._stats["recv_retries_cnt"] += 1
                if retries >= 3:
                    raise e

        e = Exception(f"Receiving message from {source} timed out")
        logger.exception(e)
        raise e

    def recv(self, source):
        """Receives data from the source rank node

        Args:
            source: rank of the sender node
        """
        assert source != self._rank
        if self._rank != 0:
            assert source == 0

        session_id = self._get_session_id(source, self._rank)
        time_start = time.time()

        # Receive number of messages
        total_packets = self._recv(session_id, source)
        if self._encryption:
            total_packets = self._encryption.decrypt(total_packets)
            total_packets = int.from_bytes(total_packets, "big")
        else:
            total_packets = int(total_packets)
        self._stats["recv_wait_time"] += time.time() - time_start

        # Receive packets
        data = b"".join([self._recv(session_id, source) for _ in range(total_packets)])
        if self._encryption:
            data = self._encryption.decrypt(data)
        data = pickle.loads(data)
        self._stats["recv_cnt"] += 1
        self._stats["recv_time"] += time.time() - time_start

        return data

    def _close(self):  # pragma: no cover
        if self._client is None:
            return
        logger.info("Closing Redis clients")
        if self._rank == 0:
            for i in range(1, self._world_size):
                self._client.delete(self._get_session_id(0, i))
        else:
            self._client.delete(self._get_session_id(self._rank, 0))

        self._client.close()
        self._client = None

    def __del__(self):
        """Close the communication channels gracefully on object dereferencing"""
        self._close()
