from abc import ABC
import torch.nn as nn

class Autoencoder(nn.Module, ABC):
    """
    Abstract base class for autoencoders.

    Subclasses must define 'encoder' and 'decoder' modules. The forward
    pass applies the encoder to input 'x' and then the decoder to reconstruct.
    """
    def __init__(self) -> None:
        """
        Initialize the Autoencoder base class without additional parameters.
        """
        super().__init__()

    def forward(self, x):
        """
        Execute a full autoencoder pass: encode input to latent space
        and then decode back to the original space.

        Parameters:
            x (Tensor): Input tensor of shape (..., input_dim).

        Returns:
            Tensor: Reconstructed tensor of same shape as input.
        """
        # Apply encoder then decoder sequentially
        return self.decoder(self.encoder(x))


''' ------------------------ Autoencoders for 3D Data ------------------------ '''
class Ae3Dto2D(Autoencoder):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
        )

class Ae3Dto2Dlinear(Autoencoder):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 2)
        self.decoder = nn.Linear(2, 3)
    
class Ae3Dto1D(Autoencoder):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 3),
        )


''' ------------------------ Autoencoders for 2D Data ------------------------ '''
class Ae2Dto1D(Autoencoder):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

class Ae2Dto1Dlinear(Autoencoder):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 1)
        self.decoder = nn.Linear(1, 2)
