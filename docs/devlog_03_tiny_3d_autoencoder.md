# Devlog 3: Teaching a small neural net to remember 3D shapes

## Goal

Prove that a tiny 3D convolutional autoencoder can compress a `32^3` occupancy grid into a
128-dimensional latent vector and reconstruct it accurately — the first learned component of
the pipeline, and the shape space every later milestone builds on.

## What I Built

- `tiny3dlatent/models/dataset.py`: `OccupancyDataset`, a PyTorch dataset over
  `data/procedural/metadata.json` that yields `(1, 32, 32, 32)` float tensors plus a shape-type
  index (saved for class conditioning in Milestone 5).
- `tiny3dlatent/models/autoencoder.py`: `Encoder3d` (three stride-2 `Conv3d` blocks with
  GroupNorm + SiLU, `32^3 -> 4^3`), a linear projection to a 128-d latent, and `Decoder3d`
  (linear + three `ConvTranspose3d` blocks back to `32^3` logits). 1.3M parameters total.
- `tiny3dlatent/models/metrics.py`: batch voxel IoU and logit thresholding.
- `tiny3dlatent/models/train_ae.py`: training CLI with config merging, device auto-select
  (MPS/CUDA/CPU), best-checkpoint tracking by val IoU, `history.json`, and an `--overfit N`
  sanity mode. Runs write into `outputs/runs/<timestamp>-autoencoder/`.
- `tiny3dlatent/models/recon_preview.py`: original-vs-reconstruction middle-slice grid.
- `configs/autoencoder.json`, plus `tests/test_autoencoder.py` (model shapes, IoU edge cases,
  dataset loading, and an end-to-end overfit training test on a generated 16^3 dataset).

## What I Studied

- 3D convolutions and transposed convolutions (`Conv3d`/`ConvTranspose3d` docs).
- `BCEWithLogitsLoss` and its `pos_weight` argument for class-imbalanced targets.
- Autoencoder bottlenecks as compressed shape descriptions (3D ShapeNets context).
- Overfitting a tiny batch as the standard first sanity check for a training loop.

## Key Idea In Plain English

Instead of storing 32,768 voxels, the encoder squeezes each shape into 128 numbers and the
decoder inflates them back. If reconstruction works, those 128 numbers are a usable "shape
description" — and generating new shapes later only requires generating 128 numbers, not a
whole grid.

## Main Result

- Overfit-8 sanity run: training loss fell to ~0.004 and IoU on the memorized batch reached
  **0.993** (`outputs/runs/20260610-160116-autoencoder-overfit/`).
- Full run (1000 train / 200 val, 40 epochs, ~10 min on MPS): best **val IoU 0.870**, above
  the 0.85 milestone threshold (`outputs/runs/20260610-160210-autoencoder/`).
- `recon_grid.png` shows all shape types recognizable after the bottleneck — cylinders,
  capsules, cubes, rounded boxes — with only mild surface smoothing.

## What Failed Or Confused Me

- The first version (no normalization, plain BCE) silently predicted *all-empty* grids for
  ~30 epochs: occupancy is only ~6% of voxels, so the loss happily decreased while IoU sat at
  exactly 0. Two fixes: GroupNorm after every conv block (much faster optimization) and
  `pos_weight=3.0` in the BCE loss to stop empty voxels from dominating. After that, the same
  overfit test went from IoU 0.55 to 0.98 in half the epochs.
- IoU is noisy between epochs even when loss falls smoothly — thresholding logits at 0.5 makes
  small logit shifts flip many border voxels at once.

## What This Teaches About Modern 3D Generation

This is the same role the sparse 3D VAE plays in TRELLIS or the geometry VAE in Step1X-3D:
a learned compressor that turns expensive spatial data into a compact latent where generation
is cheap. The class imbalance lesson also scales up — real systems care a lot about how loss
functions weight occupied vs empty space.

## Next Step

Milestone 4 turns this autoencoder into a VAE (mean/log-variance heads, KL warmup) so the
latent space becomes smooth enough to sample from and interpolate through — the property the
Milestone 5 latent generator depends on.
