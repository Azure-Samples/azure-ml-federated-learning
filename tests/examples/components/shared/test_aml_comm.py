import os
import sys
import time
import unittest
from unittest.mock import MagicMock
import multiprocessing as mp

import torch

from utils import get_free_port

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
)
from examples.components.shared.aml_comm import AMLCommSocket, AMLCommRedis
from examples.components.shared.aml_smpc import AMLSMPC


# Initialize the communication channel and afterwards test sending and receiving messages
def init_send_recv(
    rank,
    world_size,
    run_id,
    host,
    port,
    msg_send,
    msg_recv,
    shared_dict,
    encrypted=False,
    use_redis=False,
):
    shared_dict[rank] = False
    encryption = None
    if encrypted:
        encryption = AMLSMPC()

    if rank != 0:
        # Make sure the first process is started before the others
        time.sleep(1)

    if use_redis:
        comm = TestAMLCommRedis(
            shared_dict, rank, world_size, run_id, host, port, encryption=encryption
        )
    else:
        comm = AMLCommSocket(
            rank, world_size, run_id, host, port, encryption=encryption, timeout=10
        )

    if rank == 0:
        comm.send(msg_send, 1)
        result = comm.recv(1) == msg_recv
    else:
        result = comm.recv(0) == msg_recv
        comm.send(msg_send, 0)

    # Make sure the last process to send/receive message is still online for the others to send/receive
    time.sleep(1)

    print(f"Shared dict before: {shared_dict}")
    if hasattr(result, "__len__"):
        shared_dict[rank] = torch.all(result)
    else:
        shared_dict[rank] = result
    print(f"Shared dict after: {shared_dict}")


# Mock the redis connection to avoid having to run a redis server
class TestAMLCommRedis(AMLCommRedis):
    def __init__(self, buffer, *args, **kwargs):
        self.buffer = buffer
        self._get_connection_string = MagicMock(return_value="localhost")
        self._format_connection_string = MagicMock(return_value="localhost")
        self._wait_for_connection = MagicMock()
        self._connect_to_redis = MagicMock()
        super().__init__(*args, **kwargs)
        self._client = None

    def _get_session_id(self, source, destination, flag=None):
        session_id = super()._get_session_id(source, destination, flag)
        if session_id not in self.buffer:
            self.buffer[session_id] = []
        return session_id

    def _send(self, data, session_id, _) -> None:
        self.buffer[session_id] = self.buffer[session_id] + [data]

    def _recv(self, session_id, _) -> bytes:
        while len(self.buffer[session_id]) == 0:
            time.sleep(1)
        item = self.buffer[session_id][0]
        self.buffer[session_id] = self.buffer[session_id][1:]
        return item


class TestAMLComm(unittest.TestCase):
    def test_aml_comm_simple(self):
        TEST_MSG = lambda recv: f"Message to {recv}"

        for encrypted in [True, False]:
            for use_redis in [True, False]:
                with self.subTest(encrypted=encrypted, use_redis=use_redis):
                    print(
                        f"Testing: test_aml_comm_simple, encrypted: {encrypted}, use_redis: {use_redis}"
                    )
                    # Create two processes that send each other message
                    manager = mp.Manager()
                    shared_dict = manager.dict()
                    port = get_free_port()
                    p1 = mp.Process(
                        target=init_send_recv,
                        args=(
                            0,
                            2,
                            "test_run_id",
                            "localhost",
                            port,
                            TEST_MSG(1),
                            TEST_MSG(0),
                            shared_dict,
                            encrypted,
                            use_redis,
                        ),
                    )
                    p2 = mp.Process(
                        target=init_send_recv,
                        args=(
                            1,
                            2,
                            "test_run_id",
                            "localhost",
                            port,
                            TEST_MSG(0),
                            TEST_MSG(1),
                            shared_dict,
                            encrypted,
                            use_redis,
                        ),
                    )
                    p1.start()
                    p2.start()
                    p1.join()
                    p2.join()

                    for i in range(2):
                        self.assertTrue(shared_dict[i])

    def test_aml_comm_empty(self):
        TEST_MSG = lambda _: ""

        for encrypted in [True, False]:
            for use_redis in [True, False]:
                with self.subTest(encrypted=encrypted, use_redis=use_redis):
                    print(
                        f"Testing: test_aml_comm_empty, encrypted: {encrypted}, use_redis: {use_redis}"
                    )
                    # Create two processes that send each other message
                    manager = mp.Manager()
                    shared_dict = manager.dict()
                    port = get_free_port()
                    p1 = mp.Process(
                        target=init_send_recv,
                        args=(
                            0,
                            2,
                            "test_run_id",
                            "localhost",
                            port,
                            TEST_MSG(1),
                            TEST_MSG(0),
                            shared_dict,
                            encrypted,
                            use_redis,
                        ),
                    )
                    p2 = mp.Process(
                        target=init_send_recv,
                        args=(
                            1,
                            2,
                            "test_run_id",
                            "localhost",
                            port,
                            TEST_MSG(0),
                            TEST_MSG(1),
                            shared_dict,
                            encrypted,
                            use_redis,
                        ),
                    )
                    p1.start()
                    p2.start()
                    p1.join()
                    p2.join()

                    for i in range(2):
                        self.assertTrue(shared_dict[i])

    def test_aml_comm_socket_large_tensor(self):
        print(f"Testing: test_aml_comm_socket_large_tensor")

        TEST_MSG_0 = torch.randn(1000, 1000, 100)
        TEST_MSG_1 = torch.randn(1000, 1000, 100)

        manager = mp.Manager()
        shared_dict = manager.dict()
        port = get_free_port()
        p1 = mp.Process(
            target=init_send_recv,
            args=(
                0,
                2,
                "test_run_id",
                "localhost",
                port,
                TEST_MSG_0,
                TEST_MSG_1,
                shared_dict,
            ),
        )
        p2 = mp.Process(
            target=init_send_recv,
            args=(
                1,
                2,
                "test_run_id",
                "localhost",
                port,
                TEST_MSG_1,
                TEST_MSG_0,
                shared_dict,
            ),
        )
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        for i in range(2):
            self.assertTrue(shared_dict[i])


if __name__ == "__main__":
    unittest.main(verbosity=2)
