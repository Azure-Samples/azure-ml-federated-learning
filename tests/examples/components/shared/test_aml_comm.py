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
    shared_lock,
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
            shared_dict,
            shared_lock,
            rank,
            world_size,
            run_id,
            host,
            port,
            encryption=encryption,
        )
    else:
        comm = AMLCommSocket(
            rank, world_size, run_id, host, port, encryption=encryption, timeout=30
        )

    if rank == 0:
        comm.send(msg_send, 1)
        result = comm.recv(1) == msg_recv
    else:
        result = comm.recv(0) == msg_recv
        comm.send(msg_send, 0)

    if hasattr(result, "__len__"):
        shared_dict[rank] = torch.all(result)
    else:
        shared_dict[rank] = result


# Mock the redis connection to avoid having to run a redis server
class TestAMLCommRedis(AMLCommRedis):
    def __init__(self, buffer, lock, *args, **kwargs):
        self.buffer = buffer
        self.lock = lock
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
        try:
            self.lock.acquire(True)
            self.buffer[session_id] = self.buffer[session_id] + [data]
        finally:
            self.lock.release()

    def _recv(self, session_id, _) -> bytes:
        item = None
        try:
            self.lock.acquire(True)
            while len(self.buffer[session_id]) == 0:
                self.lock.release()
                time.sleep(1)
                self.lock.acquire(True)

            item = self.buffer[session_id][0]
            self.buffer[session_id] = self.buffer[session_id][1:]
        finally:
            self.lock.release()
        return item


class TestAMLComm(unittest.TestCase):
    def test_aml_comm_full(self):
        message_fns = {
            "basic": lambda recv: f"Message to {recv}",
            "empty": lambda _: f"",
            "large": lambda _: torch.randn(1000, 1000, 100),
        }

        for message_type in message_fns:
            for encrypted in [False, True]:
                for use_redis in [False, True]:
                    with self.subTest(
                        encrypted=encrypted,
                        use_redis=use_redis,
                        message_type=message_type,
                    ):
                        message_0 = message_fns[message_type](0)
                        message_1 = message_fns[message_type](1)

                        # Create two processes that send each other message
                        manager = mp.Manager()
                        shared_dict = manager.dict()
                        shared_lock = manager.Lock()
                        port = get_free_port()
                        p1 = mp.Process(
                            target=init_send_recv,
                            args=(
                                0,
                                2,
                                "test_run_id",
                                "localhost",
                                port,
                                message_1,
                                message_0,
                                shared_dict,
                                shared_lock,
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
                                message_0,
                                message_1,
                                shared_dict,
                                shared_lock,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
