import torch
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    """FraudDataset Dataset - combination of features and labels

    Args:
        df: Pandas dataframe containing features and/or labels

    Returns:
        None
    """

    def __init__(self, df):
        if "is_fraud" in df.columns:
            if len(df.columns) > 1:
                self.X = df.loc[:, df.columns != "is_fraud"].values
            else:
                self.X = None
            self.Y = df.loc[:, "is_fraud"].values
        else:
            self.X = df.values
            self.Y = None

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
