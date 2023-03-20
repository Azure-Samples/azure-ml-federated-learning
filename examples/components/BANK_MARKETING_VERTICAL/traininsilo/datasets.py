import torch
from torch.utils.data import Dataset


class BankMarketingDataset(Dataset):
    """SubscribeDataset Dataset - combination of features and labels

    Args:
        feature: Transaction detail tensors
        target: Tensor of labels corresponding to features

    Returns:
        None
    """

    def __init__(self, df):
        if "label" in df.columns:
            if len(df.columns) > 1:
                self.X = torch.tensor(
                    df.loc[:, df.columns != "label"].values, dtype=torch.float
                )
            else:
                self.X = None
            self.Y = torch.tensor(df.loc[:, "label"].values, dtype=torch.int)
        else:
            self.X = torch.tensor(df.values, dtype=torch.float)
            self.Y = None

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
