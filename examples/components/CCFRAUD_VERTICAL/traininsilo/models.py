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
