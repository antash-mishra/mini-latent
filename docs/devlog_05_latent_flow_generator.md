# Devlog 5: Generating 3D shapes in latent space instead of optimizing them one by one

## Goal

Stop reconstructing known shapes and start sampling new ones: train a small class-conditioned
generator over the VAE's 128-d latent space, then decode and mesh whatever it produces.

## What I Built

- `tiny3dlatent/models/flow.py`: `LatentFlow`, a rectified-flow velocity MLP (~200k params).
  Input is the noisy latent concatenated with a sinusoidal time embedding and a learned
  shape-class embedding; output is the predicted velocity. `sample()` Euler-integrates the
  field from N(0, I) noise to a latent in 50 steps. `rectified_flow_pair` builds the
  straight-path training pair `x_t = (1-t)*noise + t*latent`, target `latent - noise`.
- `tiny3dlatent/models/train_flow.py`: loads the frozen Milestone 4 VAE, encodes all 1000
  train grids into latent means once, standardizes them per-dimension (stats stored in the
  checkpoint), and trains the flow with flow-matching MSE. 400 epochs in ~15 seconds.
- `tiny3dlatent/models/generate_cli.py`: samples N latents per shape class, un-normalizes,
  decodes through the frozen VAE, thresholds at 0.5, then runs the Milestone 2 mesh pipeline
  on the results. Writes `generation_stats.json` with per-class checks and a per-class
  generated mesh render grid.
- `configs/latent_flow.json`, `tests/test_latent_flow.py` — including an end-to-end toy test
  that trains a flow on two synthetic latent clusters and asserts class-conditional samples
  land in the right cluster.

## What I Studied

- Flow Matching for Generative Modeling and the Rectified Flow paper — why regressing a
  constant velocity along straight noise->data paths is a valid (and simple) generative
  objective.
- Why latent-space generation is cheap: the generator never touches `32^3` grids, only
  128-d vectors, so it is an MLP instead of a 3D UNet.
- Class conditioning via embedding concatenation.

## Key Idea In Plain English

The VAE turned every training shape into a point in a 128-dimensional space. The flow model
learns a "wind field" over that space: drop a random point in, and the wind blows it toward
regions where real shapes of the requested class live. Decoding the landing point gives a
brand-new shape — no per-shape optimization, just one network pass per integration step.

## Main Result

(`outputs/runs/20260610-163025-latent-flow/` and `outputs/runs/20260610-163053-latent-generation/`)

- Flow-matching loss converges to ~0.86 (the remaining loss is the irreducible variance of
  the velocity target, not an error).
- 16 samples x 6 classes, all programmatic checks pass **96/96**: every sample is non-empty,
  a single connected component, and inside its class's plausible filled-voxel range
  (0.5x min to 1.5x max of the training distribution).
- The generated mesh grid shows clear class identity: spheres are round, cubes blocky,
  capsules elongated, toruses have genuine holes — with visible within-class variation in
  size and orientation.

## What Failed Or Confused Me

- Generated cylinders look like rounded lumps rather than crisp cylinders. I initially
  suspected the flow, but decoding the VAE reconstruction of a *real* cylinder produces the
  same rounded-prism character — the flow is at the decoder's quality ceiling. At `32^3` with
  random rotations, cylinder vs rounded-box is genuinely subtle; this is a resolution/VAE
  question (Milestone 9's territory), not a generator bug.
- Standardizing latents before flow training mattered: per-dimension scales differ enough
  that an unnormalized flow wastes capacity learning the scale instead of the structure.

## What This Teaches About Modern 3D Generation

This is the core loop of TRELLIS, TripoSG, and Step1X-3D in miniature: a frozen 3D VAE
defines the latent space, and a flow/diffusion model (theirs are DiTs, mine is a 4-layer MLP)
generates in it. The economics are the same at every scale — generation cost lives in the
latent model, decode cost is paid once at the end, and output quality is capped by the VAE.

## Next Step

Milestone 6 swaps the single class label for a tiny parsed-text condition vector (color,
size, descriptor, shape), bringing prompt-driven generation — "blue tall cylinder" — plus
optional classifier-free guidance.
