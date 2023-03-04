import torch
from torch.utils.data import Sampler


class VerticallyDistributedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, comm, rank, world_size, shuffle=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.comm = comm

    def __iter__(self):
        if self.rank == 0:
            if self.shuffle:
                indices = torch.randperm(len(self.data_source))
            else:
                indices = torch.arange(len(self.data_source))

            # Split the indices into batches
            batches = [
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]

            for batch in batches:
                for i in range(1, self.world_size):
                    # Send the batch to contributor i
                    self.comm.send(batch, i)

                yield batch
        else:
            for i in range(0, len(self.data_source), self.batch_size):
                # Receive the batch from host
                batch = self.comm.recv(0)
                yield batch

    def __len__(self):
        return torch.ceil(len(self.data_source) / self.batch_size).item()
