import os
import sys
import time
import math
import unittest
import multiprocessing as mp

import torch
from torch.utils.data import TensorDataset

from utils import get_free_port

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
)

from examples.components.shared.samplers import VerticallyDistributedBatchSampler
from examples.components.shared.aml_comm import AMLCommSocket


def init_sampler(
    rank, world_size, run_id, host, port, ds, shared_dict, batch_size, shuffle=False
):
    if rank != 0:
        # Make sure the first process is started before the others
        time.sleep(1)
    comm = AMLCommSocket(rank, world_size, run_id, host, port, timeout=10)
    sampler = VerticallyDistributedBatchSampler(
        ds, batch_size, comm, rank, world_size, shuffle=shuffle
    )
    shared_dict[rank] = ([batch for batch in sampler], len(sampler))


class TestVerticallyDistributedBatchSampler(unittest.TestCase):
    def test_vertically_distributed_batch_sampler_basic(self):
        for shuffle in [False, True]:
            with self.subTest(shuffle=shuffle):
                # Create data and set a batch size
                data = torch.randn(100, 100)
                batch_size = 30

                # Split data vertically into two datasets
                dataset1 = TensorDataset(data[:, :50])
                dataset2 = TensorDataset(data[:, 50:])

                # Create two processes that each have a sampler
                manager = mp.Manager()
                shared_dict = manager.dict()
                port = get_free_port()

                p1 = mp.Process(
                    target=init_sampler,
                    args=(
                        0,
                        2,
                        "test_run_id",
                        "localhost",
                        port,
                        dataset1,
                        shared_dict,
                        batch_size,
                        shuffle,
                    ),
                )
                p2 = mp.Process(
                    target=init_sampler,
                    args=(
                        1,
                        2,
                        "test_run_id",
                        "localhost",
                        port,
                        dataset2,
                        shared_dict,
                        batch_size,
                        shuffle,
                    ),
                )
                p1.start()
                p2.start()
                p1.join()
                p2.join()

                # Check that the number of batches is correct
                self.assertEquals(shared_dict[0][1], shared_dict[1][1])
                self.assertEquals(shared_dict[0][1], math.ceil(len(data) / batch_size))

                if shuffle:
                    # Check that the batches are distributed correctly
                    for batch1, batch2 in zip(shared_dict[0][0], shared_dict[1][0]):
                        self.assertTrue(torch.all(batch1 == batch2))
                else:
                    # Check that the batches are distributed correctly
                    for i, (batch1, batch2) in enumerate(
                        zip(shared_dict[0][0], shared_dict[1][0])
                    ):
                        self.assertTrue(torch.all(batch1 == batch2))
                        self.assertTrue(
                            torch.all(
                                batch1
                                == torch.tensor(
                                    range(
                                        i * batch_size,
                                        min((i + 1) * batch_size, len(data)),
                                    )
                                )
                            )
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
