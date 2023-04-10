import torch
import torch.nn as nn


class SimpleLinearBottom(nn.Module):
    """Bottom (Contributor) part of the model composed of only Linear model interleaved with ReLU activations

    Args:
        input_dim (int):
        number of features to be consumed by the model
    """

    def __init__(self, input_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        return self.model(x)


class SimpleLinearTop(nn.Module):
    """Top (Host) part of the model composed of only Linear model interleaved with ReLU activations"""

    def __init__(self, world_size) -> None:
        super().__init__()

        self._world_size = world_size
        self.contributor_weights = torch.nn.ModuleList(
            [nn.Linear(64, 64) for _ in range(self._world_size)]
        )

        self.model = nn.Sequential(
            nn.Linear(64, 1),
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
        agg_x = self.contributor_weights[0](x[0])
        for i in range(1, self._world_size - 1):
            agg_x += self.contributor_weights[i](x[i])

        return self.model(agg_x).squeeze()
