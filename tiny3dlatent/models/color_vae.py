from __future__ import annotations

import torch
from torch import nn

from tiny3dlatent.models.autoencoder import Decoder3d, Encoder3d


class ColorVAE(nn.Module):
    """VAE over occupancy + RGB + material grids, with separate decoders.

    Inputs are `(B, 6, R, R, R)` (occupancy, RGB, roughness, metallic), so the
    latent encodes geometry, color, and material together. A small material
    head predicts per-object `(roughness, metallic)` back out of the latent.
    """

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
        self.encoder = Encoder3d(resolution, base_channels, in_channels=6)
        self.to_mean = nn.Linear(self.encoder.feature_dim, latent_dim)
        self.to_logvar = nn.Linear(self.encoder.feature_dim, latent_dim)
        self.shape_decoder = Decoder3d(
            resolution, latent_dim, base_channels, out_channels=1
        )
        self.color_decoder = Decoder3d(
            resolution, latent_dim, base_channels, out_channels=3
        )
        self.material_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.to_mean(features), self.to_logvar(features)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Occupancy logits only, for compatibility with shape-only callers."""
        return self.shape_decoder(z)

    def decode_full(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(occupancy_logits, rgb, material)`; rgb in [0, 1]."""
        occupancy_logits = self.shape_decoder(z)
        rgb = torch.sigmoid(self.color_decoder(z))
        material = self.material_head(z)
        return occupancy_logits, rgb, material

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        occupancy_logits, rgb, material = self.decode_full(z)
        return occupancy_logits, rgb, material, mean, logvar


def masked_color_loss(
    predicted_rgb: torch.Tensor, target_rgb: torch.Tensor, occupancy: torch.Tensor
) -> torch.Tensor:
    """MSE over RGB restricted to occupied voxels (empty space has no color)."""
    mask = occupancy.expand_as(predicted_rgb)
    occupied = mask.sum().clamp_min(1.0)
    return (((predicted_rgb - target_rgb) ** 2) * mask).sum() / occupied
