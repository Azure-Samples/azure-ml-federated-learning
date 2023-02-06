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

    def __init__(self, df):
        self.X = torch.tensor(
            df.loc[:, df.columns != "is_fraud"].values, dtype=torch.float
        )
        self.Y = torch.tensor(df.loc[:, "is_fraud"].values, dtype=torch.int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is None:
            return [self.X[idx]]
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
        self.X = torch.tensor(
            df.loc[:, df.columns != "is_fraud"].values, dtype=torch.float
        )
        self.Y = torch.tensor(df.loc[:, "is_fraud"].values, dtype=torch.int)

        assert time_steps >= 10

        self._time_steps = time_steps
        self._time_step_overlaps = time_steps // 5

    def __len__(self):
        return (
            len(self.X) // (self._time_steps // self._time_step_overlaps)
            - self._time_step_overlaps
        ) + 1

    def __getitem__(self, idx):
        if self.Y is None:
            return (
                self.X[
                    idx
                    * (self._time_steps // self._time_step_overlaps) : idx
                    * (self._time_steps // self._time_step_overlaps)
                    + self._time_steps
                ],
            )
        return (
            self.X[
                idx
                * (self._time_steps // self._time_step_overlaps) : idx
                * (self._time_steps // self._time_step_overlaps)
                + self._time_steps
            ],
            self.Y[
                idx
                * (self._time_steps // self._time_step_overlaps) : idx
                * (self._time_steps // self._time_step_overlaps)
                + self._time_steps
            ],
        )
