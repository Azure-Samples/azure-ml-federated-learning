import torch
import torch.nn as nn


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
