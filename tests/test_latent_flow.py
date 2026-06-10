from __future__ import annotations

import torch

from tiny3dlatent.models.flow import (
    LatentFlow,
    rectified_flow_pair,
    time_embedding,
)


def test_time_embedding_shape_and_range() -> None:
    t = torch.tensor([0.0, 0.5, 1.0])

    embedding = time_embedding(t, 64)

    assert embedding.shape == (3, 64)
    assert torch.isfinite(embedding).all()
    assert embedding.abs().max() <= 1.0


def test_rectified_flow_pair_endpoints_and_target() -> None:
    latents = torch.randn(4, 8)
    noise = torch.randn(4, 8)

    at_zero, target = rectified_flow_pair(latents, torch.zeros(4), noise)
    at_one, _ = rectified_flow_pair(latents, torch.ones(4), noise)

    assert torch.allclose(at_zero, noise)
    assert torch.allclose(at_one, latents)
    assert torch.allclose(target, latents - noise)


def test_latent_flow_forward_and_sample_shapes() -> None:
    model = LatentFlow(latent_dim=16, num_classes=6, hidden_dim=32)
    z_t = torch.randn(5, 16)
    t = torch.rand(5)
    classes = torch.randint(0, 6, (5,))

    velocity = model(z_t, t, classes)
    samples = model.sample(classes, steps=4)

    assert velocity.shape == (5, 16)
    assert samples.shape == (5, 16)
    assert torch.isfinite(samples).all()


def test_latent_flow_learns_class_conditioned_means() -> None:
    # Two classes whose "latents" sit at +2 and -2; after training, conditional
    # samples must separate by class.
    torch.manual_seed(0)
    latent_dim = 4
    class_means = {0: 2.0, 1: -2.0}
    latents = torch.cat(
        [
            torch.full((128, latent_dim), value) + 0.1 * torch.randn(128, latent_dim)
            for value in class_means.values()
        ]
    )
    classes = torch.cat(
        [torch.zeros(128, dtype=torch.long), torch.ones(128, dtype=torch.long)]
    )

    model = LatentFlow(latent_dim=latent_dim, num_classes=2, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        t = torch.rand(latents.shape[0])
        noise = torch.randn_like(latents)
        z_t, target = rectified_flow_pair(latents, t, noise)
        loss = torch.nn.functional.mse_loss(model(z_t, t, classes), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    samples_zero = model.sample(torch.zeros(64, dtype=torch.long), steps=20)
    samples_one = model.sample(torch.ones(64, dtype=torch.long), steps=20)

    assert samples_zero.mean() > 1.0
    assert samples_one.mean() < -1.0
