from abc import ABC
import os
import math
import sys
import datetime
import time
import pickle
import logging
import socket

from enum import Enum

import mlflow
from azure.core.exceptions import ResourceNotFoundError
from azure.servicebus.exceptions import SessionLockLostError
from azure.servicebus import (
    ServiceBusClient,
    ServiceBusReceiver,
    ServiceBusSender,
    ServiceBusMessage,
)
from azure.servicebus.management import ServiceBusAdministrationClient
from azure.identity import ManagedIdentityCredential

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

    def __init__(self, rank, world_size, run_id) -> None:
        """Initializes AMLComm communicator

        Args:
            rank: rank of the current node, must be between 0 and world_size-1
            world_size: number of nodes in our setup, must be positive integer larger than 1
            run_id: specifier of the run share across the nodes
        """
        super(AMLCommSocket, self).__init__(rank, world_size, run_id)

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
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(
            6000
        )  # Set timeout so we do not wait for any request indefinitely

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
        while time.time() - time_start < 60 and tries < 3:
            try:
                msg = pickle.dumps({"flag": flag, "data": data})
                conn.sendall(msg)

                msg = conn.recv(1024)
                msg = pickle.loads(msg)
                if "flag" in msg and msg["flag"] == ok_flag:
                    break
                else:
                    logger.exception(
                        f"Message does not match ok flag {ok_flag}, received message: {msg}"
                    )
                    continue
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
        while time.time() - time_start < 60 and tries < 3:
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
                conn.setblocking(1)
                tries += 1
                time.sleep(1)
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
        if self._rank == 0:
            for c in self._connections:
                self._connections[c][0].close()

        self._socket.close()

    def __del__(self):
        """Close the communication channels gracefully on object dereferencing"""
        self.close()


class AMLCommSBAuthMethod(Enum):
    MANAGED_IDENTITY = 1
    CONNECTION_STRING = 2


class AMLCommSB(AMLComm):
    """AMLCommSB provides simple communication layer across nodes in AzureML
    using ServiceBus. The communication capabilities are limited to scatter-gather
    pattern, where node 0 is considered as a host(master) node. Communication
    channel is established over Python sockets and data are serialized
    using pickle library.
    """

    READY_TOKEN_MESSAGE = "READY"

    def __init__(
        self,
        rank: int,
        world_size: int,
        run_id: str,
        auth_method: AMLCommSBAuthMethod,
        sb_namespace: str = None,
        connection_string: str = None,
        queue_discovery_timeout=0,
    ) -> None:
        """Initializes AMLCommSB communicator

        Args:
            rank: rank of the current node, must be between 0 and world_size-1
            world_size: number of nodes in our setup, must be positive integer larger than 1
            run_id: specifier of the run share across the nodes
        """
        super(AMLCommSB, self).__init__(rank, world_size, run_id)

        assert (
            auth_method == AMLCommSBAuthMethod.MANAGED_IDENTITY
            and sb_namespace is not None
        ) or (
            auth_method == AMLCommSBAuthMethod.CONNECTION_STRING
            and connection_string is not None
        )

        self._sb_namespace = sb_namespace
        self._auth_method = auth_method
        self._connection_string = connection_string
        self._queue_discovery_timeout = queue_discovery_timeout

        self._senders: dict[str, ServiceBusSender] = {}
        self._receivers: dict[str, ServiceBusReceiver] = {}

        self._authenticate()
        self._client: ServiceBusClient = self._initialize_client()
        self._admin_client: ServiceBusAdministrationClient = (
            self._initialize_admin_client()
        )

        if self._rank == 0:
            # Let server create the queue to be used for communication
            # self._admin_client.create_queue(self._run_id, requires_session=True, auto_delete_on_idle=datetime.timedelta(seconds=300))
            self._init_server()
        else:
            self._init_client()

    def _init_server(self):
        if self._run_id not in [q["name"] for q in self._admin_client.list_queues()]:
            self._admin_client.create_queue(
                self._run_id,
                requires_session=True,
                auto_delete_on_idle=datetime.timedelta(seconds=300),
            )

        for destination in range(1, self._world_size):
            session_id = self._get_session_id(destination, self._rank)
            self._receivers[session_id] = self._client.get_queue_receiver(
                self._run_id, session_id=session_id
            )
            session_id = self._get_session_id(self._rank, destination)
            self._senders[session_id] = self._client.get_queue_sender(
                self._run_id, session_id=session_id
            )

            message_size = ServiceBusMessage(
                self.READY_TOKEN_MESSAGE, session_id=session_id
            )
            self._senders[session_id].send_messages(message_size)

    def _init_client(self):
        start_time = time.time()
        session_id = self._get_session_id(self._rank, 0)
        self._senders[session_id] = self._client.get_queue_sender(
            self._run_id, session_id=session_id
        )

        session_id = self._get_session_id(0, self._rank)
        self._receivers[session_id] = self._client.get_queue_receiver(
            self._run_id, session_id=session_id
        )

        while True:
            try:
                # Make sure that the queue exists
                self._admin_client.get_queue(self._run_id)

                # Receive ready messages
                message = self._receivers[session_id].receive_messages()[0]
                self._receivers[session_id].complete_message(message)
                if str(message) == self.READY_TOKEN_MESSAGE:
                    return
            except ResourceNotFoundError as e:
                logger.warning(
                    f"Requested queue '{self._run_id}' does not exist, waiting..."
                )
                time.sleep(10)

                if (
                    self._queue_discovery_timeout > 0
                    and time.time() - start_time > self._queue_discovery_timeout
                ):
                    raise e
            except SessionLockLostError:
                self._receivers[session_id].close()
                self._receivers[session_id] = self._client.get_queue_receiver(
                    queue_name=self._run_id, session_id=session_id
                )
            except Exception as e:
                logger.warning(f"Message does not contain start token, receiving...")
                if (
                    self._queue_discovery_timeout > 0
                    and time.time() - start_time > self._queue_discovery_timeout
                ):
                    raise e

    def _authenticate(self):
        if self._auth_method == AMLCommSBAuthMethod.MANAGED_IDENTITY:
            if "DEFAULT_IDENTITY_CLIENT_ID" in os.environ:
                self.logger.info(
                    "Using default identity client id {}".format(
                        os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
                    )
                )
                self._credential = ManagedIdentityCredential(
                    client_id=os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
                )
            else:
                self._credential = ManagedIdentityCredential()
        elif self._auth_method == AMLCommSBAuthMethod.CONNECTION_STRING:
            if self._connection_string is not None:
                return
            elif "AMLCOMMSB_CONNECTION_STRING" in os.environ:
                self._connection_string = os.environ.get("AMLCOMMSB_CONNECTION_STRING")
            else:
                from azureml.core import Run

                run = Run.get_context()
                ws = run.experiment.workspace
                kv = ws.get_default_keyvault()
                self._connection_string = kv.get_secret("AMLCOMMSB_CONNECTION_STRING")
        else:
            raise Exception("Unknown auth_method {}".format(self._auth_method))

    def _initialize_client(self) -> ServiceBusClient:
        if self._auth_method == AMLCommSBAuthMethod.CONNECTION_STRING:
            return ServiceBusClient.from_connection_string(self._connection_string)
        elif self._auth_method == AMLCommSBAuthMethod.MANAGED_IDENTITY:
            return ServiceBusClient(
                self._sb_namespace,
                self._credential,
            )

    def _initialize_admin_client(self) -> ServiceBusAdministrationClient:
        if self._auth_method == AMLCommSBAuthMethod.CONNECTION_STRING:
            return ServiceBusAdministrationClient.from_connection_string(
                self._connection_string
            )
        elif self._auth_method == AMLCommSBAuthMethod.MANAGED_IDENTITY:
            return ServiceBusAdministrationClient(
                self._sb_namespace,
                self._credential,
            )

    def _get_session_id(self, source: int, destination: int, flag: FLAGS = None):
        if flag is None:
            return f"{source}=>{destination}"
        return f"{source}=>{destination}:{FLAGS[flag].name}"

    def _send(self, session_id, data):
        try:
            message = ServiceBusMessage(data, session_id=session_id)
            self._senders[session_id].send_messages(message)
        except Exception as e:
            logger.exception(f"Sending failed with exception:{e}")
            raise e

    def send(self, data, destination) -> None:
        session_id = self._get_session_id(self._rank, destination)
        message = pickle.dumps(data)

        # Message needs to be split into packets smaller than 256KB,
        # in our case the target is 250KB due to ServiceBus envelope
        total_packets = math.ceil(sys.getsizeof(message) / (250 * 1024))
        packet_size = len(message) // total_packets
        time_start = time.time()

        # Inform destination about number of packets to be received
        self._send(session_id, str(total_packets))

        # Send the packets one by one, these packets are usually too large to send them in batches
        for i in range(total_packets):
            packet = (
                message[i * packet_size : (i + 1) * packet_size]
                if i < total_packets - 1
                else message[i * packet_size :]
            )
            self._send(session_id, packet)

        self._stats["msg_sent"] += 1
        self._stats["sending_time"] += time.time() - time_start

    def _recv(self, session_id, total_packets=1):
        message = []
        received_packets = 0
        while received_packets < total_packets:
            try:
                packets = self._receivers[session_id].receive_messages(
                    total_packets - received_packets
                )
                for packet in packets:
                    message.extend([seq for seq in packet.body])
                    # message += b"".join(seq for seq in packet.body)
                    self._receivers[session_id].complete_message(packet)
                    received_packets += 1
            except SessionLockLostError:
                self._receivers[session_id].close()
                self._receivers[session_id] = self._client.get_queue_receiver(
                    queue_name=self._run_id, session_id=session_id
                )
            except Exception as e:
                logger.exception(f"Receiving failed with exception:{e}")
                raise e
        return message

    def recv(self, source):
        session_id = self._get_session_id(source, self._rank)
        time_start = time.time()

        message = self._recv(session_id, 1)
        total_packets = int(message[0])
        self._stats["waiting_time"] += time.time() - time_start

        message = self._recv(session_id, total_packets)

        try:
            data = b"".join(packet for packet in message)
            data = pickle.loads(data)
        except Exception as e:
            logger.exception(f"Failed to load received message: {message}")
            logger.exception(f"Failed to load received data: {data}")
            raise e

        self._stats["msg_received"] += 1
        self._stats["receiving_time"] += time.time() - time_start

        return data

    def __del__(self):
        logger.info("Closing ServiceBus clients")
        self._client.close()
        self._admin_client.close()
