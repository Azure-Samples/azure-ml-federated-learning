import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVAEBottom(nn.Module):
    """Bottom (Contributor) part of the VAE with head composed of Linear layers interleaved by ReLU activations

    Args:
        input_dim (int): number of features to be consumed by the model
        hidden_dim (int): number of hidden units in the hidden layers of the model (default: 128)
        latent_dim (int): number of latent units in the model (default: 32)
        num_layers (int): number of hidden layers in the model (default: 2)

    Note:
        Input must be 3D such that it contains time-dependent sequences
    """

    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_layers=2) -> None:
        super(SimpleVAEBottom, self).__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim)
                if i == 0
                else nn.Linear(hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )
        self.mean = torch.nn.Linear(
            in_features=self._hidden_dim,
            out_features=self._latent_dim,
        )
        self.log_variance = torch.nn.Linear(
            in_features=self._hidden_dim,
            out_features=self._latent_dim,
        )

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [
                nn.Linear(latent_dim, hidden_dim)
                if i == 0
                else (
                    nn.Linear(hidden_dim, input_dim)
                    if i == num_layers
                    else nn.Linear(hidden_dim, hidden_dim)
                )
                for i in range(num_layers + 1)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def kl_loss(self, mu, log_var):
        # KL Divergence
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl = kl.sum(-1)
        kl = kl.mean()
        return kl

    def encoder(self, x):
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            x = F.relu(layer(x))
        mu = self.mean(x)
        log_variance = self.log_variance(x)
        std = torch.exp(0.5 * log_variance)

        # Generate a unit gaussian noise.
        noise = torch.randn(*x.shape[:-1], self._latent_dim).to(x.device)
        z = noise * std + mu

        return z, mu, log_variance

    def decoder(self, x):
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            if i == len(self.decoder_layers) - 1:
                x = torch.sigmoid(layer(x))
            else:
                x = F.relu(layer(x))
        return x

    def forward(self, x):
        # Encoder
        z, mu, log_variance = self.encoder(x)
        x_hat = self.decoder(z)

        # Compute architecture loss (difference between input/output + difference between predicted distribution and Gaussian distribution)
        reconstruction_loss = F.mse_loss(x_hat.reshape(-1), x.reshape(-1))
        kl_loss = self.kl_loss(mu, log_variance)
        y_hat = z

        return y_hat, (reconstruction_loss, kl_loss)
