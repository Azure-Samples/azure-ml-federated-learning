import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinear(nn.Module):
    """Model composed of only Linear model interleaved with ReLU activations

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
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x).squeeze(), None


class SimpleLSTM(nn.Module):
    """Model composed of LSTM layers along with head composed of Linear layers interleaved by ReLU activations

    Args:
        input_dim (int):
        number of features to be consumed by the model

    Note:
        Input must be 3D such that it contains time-dependent sequences
    """

    def __init__(self, input_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=0.5)
        self.denseseq = nn.Sequential(
            nn.Linear(256, 128),
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
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.denseseq(x).squeeze()
        return x, None


class SimpleVAE(nn.Module):
    """LSTM based VAE with head composed of Linear layers interleaved by ReLU activations

    Args:
        input_dim (int):
        number of features to be consumed by the model

    Note:
        Input must be 3D such that it contains time-dependent sequences
    """

    def __init__(self, input_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self._hidden_dim = 128
        self._num_layers = 2
        self._bidirectional = False
        self._hidden_factor = (2 if self._bidirectional else 1) * self._num_layers
        self._latent_dim = 256
        self._embedding_dropout = 0.5

        # Encoder Part
        self.encoder_lstm = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self._hidden_dim,
            batch_first=True,
            num_layers=self._num_layers,
            bidirectional=self._bidirectional,
        )
        self.mean = torch.nn.Linear(
            in_features=self._hidden_dim * self._num_layers,
            out_features=self._latent_dim,
        )
        self.log_variance = torch.nn.Linear(
            in_features=self._hidden_dim * self._num_layers,
            out_features=self._latent_dim,
        )

        # Decoder part
        self.init_hidden_decoder = torch.nn.Linear(
            in_features=self._latent_dim,
            out_features=self._hidden_dim * self._num_layers,
        )
        self.embedding_dropout = nn.Dropout(p=self._embedding_dropout)
        self.decoder_lstm = torch.nn.LSTM(
            input_size=self._latent_dim,
            hidden_size=self.input_dim,
            batch_first=True,
            num_layers=self._num_layers,
            bidirectional=self._bidirectional,
        )

        self.nll_loss = torch.nn.NLLLoss(reduction="sum")
        self.output = torch.nn.Linear(
            in_features=self._latent_dim,  # * self._num_layers,
            out_features=1,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def encoder(self, x, hidden_encoder):
        output_encoder, hidden_encoder = self.encoder_lstm(x, hidden_encoder)
        hidden_state = hidden_encoder[0].permute((1, 0, 2)).reshape(x.shape[0], -1)

        mean = self.mean(hidden_state)
        log_var = self.log_variance(hidden_state)
        std = torch.exp(0.5 * log_var)

        # Generate a unit gaussian noise.
        batch_size = output_encoder.size(0)
        noise = torch.randn(batch_size, self._latent_dim).to(hidden_state.device)
        z = noise * std + mean

        return z, mean, log_var, hidden_state

    def decoder(self, z, seq_dim):
        decoder_input = z.repeat([seq_dim, 1, 1]).permute((1, 0, 2))

        # decoder forward pass
        output_decoder, _ = self.decoder_lstm(decoder_input)

        return output_decoder

    def kl_loss(self, mu, log_var):
        # KL Divergence
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl = kl.sum(-1)
        kl = kl.mean()
        return kl

    def forward(self, x):
        # Get Embeddings
        hidden_encoder = None

        # Encoder
        z, mu, log_var, hidden_encoder = self.encoder(x, hidden_encoder)

        # Decoder
        x_hat = self.decoder(z, x.shape[1])

        # Compute architecture loss (difference between input/output + difference between predicted distribution and Gaussian distribution)
        reconstruction_loss = F.mse_loss(x_hat.reshape(-1), x.reshape(-1))
        kl_loss = self.kl_loss(mu, log_var)

        y_hat = self.output(z.repeat([x.shape[1], 1, 1]).permute((1, 0, 2))).squeeze()
        y_hat = self.sigmoid(y_hat)

        return y_hat, reconstruction_loss + kl_loss
