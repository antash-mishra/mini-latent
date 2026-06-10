from __future__ import annotations

import torch
from torch import nn


class Encoder3d(nn.Module):
    """Three stride-2 conv blocks: `(B, 1, R, R, R)` -> flat conv features."""

    def __init__(self, resolution: int, base_channels: int) -> None:
        super().__init__()
        if resolution % 8 != 0:
            raise ValueError("resolution must be divisible by 8")
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.conv = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, c1),
            nn.SiLU(),
            nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, c2),
            nn.SiLU(),
            nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, c3),
            nn.SiLU(),
        )
        self.spatial = resolution // 8
        self.feature_dim = c3 * self.spatial**3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).flatten(start_dim=1)


class Decoder3d(nn.Module):
    """Latent vector -> occupancy logits of shape `(B, 1, R, R, R)`."""

    def __init__(self, resolution: int, latent_dim: int, base_channels: int) -> None:
        super().__init__()
        if resolution % 8 != 0:
            raise ValueError("resolution must be divisible by 8")
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.spatial = resolution // 8
        self.channels = c3
        self.project = nn.Linear(latent_dim, c3 * self.spatial**3)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, c2),
            nn.SiLU(),
            nn.ConvTranspose3d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, c1),
            nn.SiLU(),
            nn.ConvTranspose3d(c1, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, c1),
            nn.SiLU(),
            nn.Conv3d(c1, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(-1, self.channels, self.spatial, self.spatial, self.spatial)
        return self.deconv(x)


class Autoencoder(nn.Module):
    """Tiny 3D conv autoencoder over binary occupancy grids."""

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
        self.to_latent = nn.Linear(self.encoder.feature_dim, latent_dim)
        self.decoder = Decoder3d(resolution, latent_dim, base_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z
