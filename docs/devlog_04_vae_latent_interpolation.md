# Devlog 4: From memorizing shapes to sampling a 3D latent space

## Goal

Upgrade the Milestone 3 autoencoder to a VAE so the 128-d latent space stops being a lookup
table and becomes a smooth, sampleable space — the property the Milestone 5 latent generator
needs.

## What I Built

- `tiny3dlatent/models/vae.py`: `VAE` reusing the autoencoder's `Encoder3d`/`Decoder3d`
  trunks, with separate mean and log-variance heads, the reparameterization trick, and a
  `kl_per_dimension` helper (per-dim KL in nats, used both for the loss and for collapse
  diagnostics).
- `tiny3dlatent/models/train_vae.py`: training CLI with loss = BCE(`pos_weight=3`) +
  `beta * KL`, where beta warms up linearly over the first 10 epochs to `kl_weight=1e-4`.
  Every epoch logs val IoU from *mean* latents, val IoU from *sampled* latents, total KL,
  and the number of active latent dimensions (per-dim KL > 0.01 nats).
- `tiny3dlatent/models/interpolate_cli.py`: encodes two val shapes from different classes,
  linearly interpolates the latent means over 8 steps, decodes each step, and renders an
  interpolation strip of middle slices.
- `configs/vae.json`, `tests/test_vae.py` (shapes, KL closed-form values, reparameterization).

## What I Studied

- Auto-Encoding Variational Bayes (Kingma & Welling) — the ELBO as reconstruction + KL.
- Posterior collapse and why KL warmup helps at small scale.
- Why latent-diffusion-style systems use a *small* KL weight: the latent space only needs to
  be smooth and roughly unit-scale, since a generator will model its true distribution anyway.

## Key Idea In Plain English

The autoencoder maps each shape to one exact point; the space between points is undefined.
The VAE maps each shape to a small cloud and pays a penalty (KL) for clouds that drift far
from the origin or shrink to points. Overlapping clouds force the space between shapes to
decode into sensible shapes too — which is what makes interpolation and sampling work.

## Main Result

(`outputs/runs/20260610-161522-vae/`, ~11 min on MPS, 1.8M parameters)

- Val IoU from mean latents: **0.840** vs 0.870 for the plain AE — the expected small drop.
- Val IoU from sampled latents: **0.814** — reconstructions survive posterior noise.
- KL not collapsed: total 117 nats, **all 128 dims active** (per-dim mean 0.92, max 3.99).
- Interpolation strip (`outputs/runs/20260610-162631-vae-interpolation/`): cube -> cylinder
  rounds gradually, sphere -> cube squares off, torus -> capsule shrinks smoothly; every
  intermediate frame is a plausible shape, none are noise or empty.

## What Failed Or Confused Me

- Picking the KL weight scale was the only real decision: BCE is a mean over 32,768 voxels
  while KL is a sum over 128 dims, so beta=1 would make KL ~100x larger than the recon term
  and collapse everything to the prior instantly. `1e-4` keeps the KL contribution around
  10-30% of the loss, which regularized without collapsing.
- "Active dims = 128" initially looked suspicious (too healthy). It is a consequence of the
  small beta — with a stronger beta the model would shut unused dims. For this project's
  purpose (smoothness for a downstream generator), that trade is fine.

## What This Teaches About Modern 3D Generation

This is the same design point TRELLIS and Step1X-3D sit at: their 3D VAEs are KL-regularized
just enough to give the diffusion/flow stage a well-behaved latent space, not to make the
prior itself a good generator. Sampling straight from N(0, I) here still gives mushy blobs —
exactly why Milestone 5 trains a real generator over the latents.

## Next Step

Milestone 5: freeze this VAE, encode the train set into latent vectors, and train a small
class-conditioned rectified-flow MLP that turns noise into latents — then decode and mesh
the results.
