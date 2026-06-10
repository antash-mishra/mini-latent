from __future__ import annotations

import torch
from torch import nn

from tiny3dlatent.models.autoencoder import Decoder3d, Encoder3d


class VAE(nn.Module):
    """Tiny 3D conv VAE: the autoencoder trunks with mean/log-variance heads."""

    def __init__(
        self,
        *,
        resolution: int = 32,
        latent_dim: int = 128,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.encoder = Encoder3d(resolution, base_channels)
        self.to_mean = nn.Linear(self.encoder.feature_dim, latent_dim)
        self.to_logvar = nn.Linear(self.encoder.feature_dim, latent_dim)
        self.decoder = Decoder3d(resolution, latent_dim, base_channels)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.to_mean(features), self.to_logvar(features)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar, z


def kl_per_dimension(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) per latent dimension, averaged over the batch.

    Returns a `(latent_dim,)` tensor in nats; sum it for the total KL term.
    """
    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
    return kl.mean(dim=0)
