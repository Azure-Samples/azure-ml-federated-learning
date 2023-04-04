import numpy as np
import torch
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    """FraudDataset Dataset - combination of features and labels

    Args:
        feature: Transaction detail tensors
        target: Tensor of labels corresponding to features

    Returns:
        None
    """

    def __init__(self, df, **kwargs):
        if "is_fraud" in df.columns:
            if len(df.columns) > 1:
                self.X = df.loc[:, df.columns != "is_fraud"].values
            else:
                self.X = None
            self.Y = df.loc[:, "is_fraud"].values
        else:
            self.X = df.values
            self.Y = None

        if "embeddings" in kwargs and len(kwargs["embeddings"]) > 0:
            self.X = np.load(kwargs["embeddings"][0])
            for embedding in kwargs["embeddings"][1:]:
                np_embeddings = np.load(embedding)
                self.X = np.concatenate([self.X, np_embeddings], axis=1)

        if self.X is not None:
            self.X = torch.tensor(self.X, dtype=torch.float)
        if self.Y is not None:
            self.Y = torch.tensor(self.Y, dtype=torch.int)

    def __len__(self):
        if self.Y is None:
            return len(self.X)
        else:
            return len(self.Y)

    def features_count(self):
        if self.X is not None:
            return self.X.shape[1]
        return None

    def __getitem__(self, idx):
        if self.Y is None:
            return self.X[idx]
        elif self.X is None:
            return self.Y[idx]
        else:
            return self.X[idx], self.Y[idx]


class FraudTimeDataset(Dataset):
    """FraudTimeDataset Dataset - combination of features and labels retrieved sequentially

    Args:
        feature: Transaction detail tensors
        target: Tensor of labels corresponding to features

    Returns:
        None
    """

    def __init__(self, df, time_steps=100):
        if "is_fraud" in df.columns:
            if len(df.columns) > 1:
                self.X = torch.tensor(
                    df.loc[:, df.columns != "is_fraud"].values, dtype=torch.float
                )
            else:
                self.X = None
            self.Y = torch.tensor(df.loc[:, "is_fraud"].values, dtype=torch.int)
        else:
            self.X = torch.tensor(df.values, dtype=torch.float)
            self.Y = None

        assert time_steps >= 10

        self._time_steps = time_steps
        self._time_step_overlaps = time_steps // 5

    def __len__(self):
        if self.Y is None:
            return (
                len(self.X) // (self._time_steps // self._time_step_overlaps)
                - self._time_step_overlaps
            ) + 1
        else:
            return (
                len(self.Y) // (self._time_steps // self._time_step_overlaps)
                - self._time_step_overlaps
            ) + 1

    def features_count(self):
        if self.X is not None:
            return self.X.shape[1]
        return None

    def __getitem__(self, idx):
        left = idx * (self._time_steps // self._time_step_overlaps)
        right = idx * (self._time_steps // self._time_step_overlaps) + self._time_steps
        if self.Y is None:
            return self.X[left:right]
        elif self.X is None:
            return self.Y[left:right]
        else:
            return (self.X[left:right], self.Y[left:right])
