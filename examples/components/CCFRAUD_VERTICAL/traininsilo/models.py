import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinearBottom(nn.Module):
    """Bottom (Contributor) part of the model composed of only Linear model interleaved with ReLU activations

    Args:
        input_dim (int):
        number of features to be consumed by the model
    """

    def __init__(self, input_dim, latent_dim=4, hidden_dim=128, layers=4) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim)
                if i == 0
                else (
                    nn.Linear(hidden_dim, latent_dim)
                    if i == layers - 1
                    else nn.Linear(hidden_dim, hidden_dim)
                )
                for i in range(layers)
            ]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = F.relu(layer(x))
        return x


class SimpleLinearTop(nn.Module):
    """Top (Host) part of the model composed of only Linear model interleaved with ReLU activations"""

    def __init__(self, latent_dim) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x).squeeze()


class SimpleLSTMBottom(nn.Module):
    """Bottom (Contributor) part of the model composed of LSTM layers along with head composed of Linear layers interleaved by ReLU activations

    Args:
        input_dim (int):
        number of features to be consumed by the model

    Note:
        Input must be 3D such that it contains time-dependent sequences
    """

    def __init__(self, input_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = 256
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=0.5)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x


class SimpleLSTMTop(nn.Module):
    """Top (Host) part of the model composed of LSTM layers along with head composed of Linear layers interleaved by ReLU activations"""

    def __init__(self, latent_dim) -> None:
        super().__init__()

        self.denseseq = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.denseseq(x).squeeze()
        return x


class SimpleVAETop(nn.Module):
    """Top (Host) part of the LSTM based VAE with head composed of Linear layers interleaved by ReLU activations"""

    def __init__(self, latent_dim, hidden_dim=32) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim

        self.seq = torch.nn.Sequential(
            nn.Linear(in_features=self._latent_dim, out_features=self._hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self._hidden_dim, out_features=1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.seq(x).squeeze()
        return x
