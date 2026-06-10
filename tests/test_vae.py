from __future__ import annotations

import torch

from tiny3dlatent.models.vae import VAE, kl_per_dimension


def test_vae_shapes() -> None:
    model = VAE(resolution=16, latent_dim=32, base_channels=8)
    grids = torch.zeros(2, 1, 16, 16, 16)

    logits, mean, logvar, z = model(grids)

    assert logits.shape == (2, 1, 16, 16, 16)
    assert mean.shape == (2, 32)
    assert logvar.shape == (2, 32)
    assert z.shape == (2, 32)


def test_kl_per_dimension_is_zero_for_standard_normal_posterior() -> None:
    mean = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)

    kl = kl_per_dimension(mean, logvar)

    assert kl.shape == (8,)
    assert torch.allclose(kl, torch.zeros(8), atol=1e-6)


def test_kl_per_dimension_known_value() -> None:
    # KL for mean=1, logvar=0 is 0.5 * (1 + 1 - 1 - 0) = 0.5 per dimension.
    mean = torch.ones(3, 4)
    logvar = torch.zeros(3, 4)

    kl = kl_per_dimension(mean, logvar)

    assert torch.allclose(kl, torch.full((4,), 0.5), atol=1e-6)


def test_reparameterize_matches_mean_when_variance_is_tiny() -> None:
    model = VAE(resolution=16, latent_dim=8, base_channels=8)
    mean = torch.randn(5, 8)
    logvar = torch.full((5, 8), -40.0)

    z = model.reparameterize(mean, logvar)

    assert torch.allclose(z, mean, atol=1e-4)
