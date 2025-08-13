import torch
import torch.nn as nn

class AE(nn.Module):
    """
    Autoencoder (AE) with optional hidden layers architecture.

    Args:
        input_dim   (int): Dimensionality of the input features.
        hidden_dims (list[int]): List of hidden layer sizes. If empty, uses a single linear layer for encoding.
        latent_dim  (int): Dimensionality of the latent (bottleneck) representation.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initialize the Autoencoder model.

        Builds the encoder and decoder networks based on the hidden_dims parameter.

        If hidden_dims is empty, a single linear transformation is used for both encoder and decoder.
        Otherwise, a multi-layer MLP with SELU activations is constructed for all.
        """
        super(AE, self).__init__()
        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim

        if not hidden_dims:  # No hidden layers, linear AE
            encoder_layers.append(nn.Linear(input_dim, latent_dim))
        else:
            for h_dim in hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, h_dim))
                encoder_layers.append(nn.SELU())
                prev_dim = h_dim
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder (reverse hidden_dims)
        decoder_layers = []
        prev_dim = latent_dim

        if not hidden_dims:
            decoder_layers.append(nn.Linear(latent_dim, input_dim))
        else:
            for h_dim in reversed(hidden_dims):
                decoder_layers.append(nn.Linear(prev_dim, h_dim))
                decoder_layers.append(nn.SELU())
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """
        Encode the input into the latent representation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim).
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode the latent representation back to the input space.

        Args:
            z (torch.Tensor): Latent tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, input_dim).
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            x_recon (torch.Tensor): Reconstructed input of shape (batch_size, input_dim).
            h (torch.Tensor): Latent representation of shape (batch_size, latent_dim).
            None, None: Placeholders for compatibility with VAE interface.
        """
        h = self.encode(x)
        x_recon = self.decode(h)
        return x_recon, h, None, None


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with Gaussian latent space.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Size of the hidden layers in the encoder and decoder.
        latent_dim (int): Dimensionality of the latent (bottleneck) representation.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize the VAE model.

        Constructs the encoder network to output the mean and log-variance for the latent distribution,
        and a decoder network to reconstruct inputs from latent samples.
        """
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        """
        Encode the input into parameters of the latent Gaussian distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            mu (torch.Tensor): Mean of the latent Gaussian of shape (batch_size, latent_dim).
            logvar (torch.Tensor): Log-variance of the latent Gaussian of shape (batch_size, latent_dim).
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from a Gaussian distribution.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            logvar (torch.Tensor): Log-variance of the latent Gaussian.

        Returns:
            torch.Tensor: Sampled latent vector of shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, h):
        """
        Decode the latent representation to reconstruct the input.

        Args:
            h (torch.Tensor): Latent tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, input_dim).
        """
        return self.decoder(h)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            x_recon (torch.Tensor): Reconstructed input of shape (batch_size, input_dim).
            h (torch.Tensor): Sampled latent representation of shape (batch_size, latent_dim).
            mu (torch.Tensor): Mean of the latent Gaussian.
            logvar (torch.Tensor): Log-variance of the latent Gaussian.
        """
        mu, logvar = self.encode(x)
        h = self.reparameterize(mu, logvar)
        x_recon = self.decode(h)
        return x_recon, h, mu, logvar
