from __future__ import annotations

import math

import torch
from torch import nn


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of flow time `t` in [0, 1], shape `(B,) -> (B, dim)`."""
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    angles = t[:, None] * frequencies[None, :] * 1000.0
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class LatentFlow(nn.Module):
    """Class-conditioned rectified-flow velocity field over VAE latents.

    Predicts the constant velocity `x1 - x0` of the straight path
    `x_t = (1 - t) * x0 + t * x1` from noise `x0` to a data latent `x1`.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        num_classes: int = 6,
        hidden_dim: int = 256,
        time_dim: int = 64,
        class_dim: int = 64,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.class_embedding = nn.Embedding(num_classes, class_dim)
        layers: list[nn.Module] = [
            nn.Linear(latent_dim + time_dim + class_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, class_indices: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            [
                z_t,
                time_embedding(t, self.time_dim),
                self.class_embedding(class_indices),
            ],
            dim=1,
        )
        return self.network(features)

    @torch.no_grad()
    def sample(
        self,
        class_indices: torch.Tensor,
        *,
        steps: int = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler-integrate the learned velocity field from noise to latents."""
        device = next(self.parameters()).device
        count = class_indices.shape[0]
        z = torch.randn(count, self.latent_dim, device=device, generator=generator)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((count,), step * dt, device=device)
            z = z + self.forward(z, t, class_indices) * dt
        return z


def rectified_flow_pair(
    latents: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(x_t, velocity_target)` for the straight noise->latent path."""
    t_column = t[:, None]
    x_t = (1.0 - t_column) * noise + t_column * latents
    return x_t, latents - noise


class ConditionedLatentFlow(nn.Module):
    """Rectified flow conditioned on multiple prompt attributes.

    Each attribute (shape type, color, size, descriptor) gets its own embedding
    table with one extra "unspecified" row at index `vocab_size`, used both for
    prompts that omit the attribute and for classifier-free guidance dropout.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        attribute_sizes: tuple[int, ...] = (6, 6, 3, 5),
        attribute_dim: int = 32,
        hidden_dim: int = 256,
        time_dim: int = 64,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.attribute_sizes = tuple(attribute_sizes)
        self.embeddings = nn.ModuleList(
            nn.Embedding(size + 1, attribute_dim) for size in attribute_sizes
        )
        condition_dim = attribute_dim * len(attribute_sizes)
        layers: list[nn.Module] = [
            nn.Linear(latent_dim + time_dim + condition_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def null_attributes(self, count: int, device: torch.device) -> torch.Tensor:
        """`(count, A)` attribute indices that are all "unspecified"."""
        nulls = torch.tensor(self.attribute_sizes, device=device)
        return nulls.expand(count, -1).contiguous()

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, attributes: torch.Tensor
    ) -> torch.Tensor:
        embedded = [
            embedding(attributes[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        features = torch.cat([z_t, time_embedding(t, self.time_dim), *embedded], dim=1)
        return self.network(features)

    @torch.no_grad()
    def sample(
        self,
        attributes: torch.Tensor,
        *,
        steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Euler-integrate from noise, optionally with classifier-free guidance."""
        device = next(self.parameters()).device
        count = attributes.shape[0]
        z = torch.randn(count, self.latent_dim, device=device)
        nulls = self.null_attributes(count, device)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((count,), step * dt, device=device)
            velocity = self.forward(z, t, attributes)
            if guidance_scale != 1.0:
                unconditional = self.forward(z, t, nulls)
                velocity = unconditional + guidance_scale * (velocity - unconditional)
            z = z + velocity * dt
        return z
