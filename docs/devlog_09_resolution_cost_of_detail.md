# Devlog 9: The cost of detail in 3D generation

## Goal

Move from `32^3` to `64^3` dense voxels and *measure* what detail costs, instead of
guessing — then check whether the extra resolution fixes the small-feature failures
Milestone 8's report surfaced.

## What I Built

- `configs/procedural_dataset_64.json`: a 1000/200 dataset at `64^3` generated with the
  same seed as the `32^3` one, so example i has the *identical shape parameters* at both
  resolutions — clean matched-pairs comparison.
- `tiny3dlatent/models/benchmark_resolution.py`: times warm training steps and records
  device memory for both settings on the same hardware.
- `tiny3dlatent/models/resolution_compare_cli.py`: reconstructs matched val examples with
  the latest 32 and 64 checkpoints and renders them side by side with per-example IoU.
- A full `64^3` VAE training run (30 epochs) using the existing `train_vae.py` — the
  architecture was already resolution-parametric, only the config changed.

## What I Studied

- Where dense-voxel cost actually lives: activations scale with `R^3`, but the parameter
  blow-up is in the flatten-to-latent linear layers (4^3 x 64ch -> 8^3 x 64ch input).
- Why IoU is not comparable across resolutions: finer voxels make every surface
  disagreement cost more, so the same physical error reads as a lower score.
- Why sparse/triplane representations exist (the measured numbers below are the argument).

## Key Idea In Plain English

Doubling resolution is not "2x the work" — it is 8x the voxels. The benchmark made it
concrete: every training example costs 8x more time, and the model needed 7x more
parameters just to connect the bigger grid to the same 128-dim latent.

## Main Result

Measured cost table (`outputs/runs/20260611-094546-resolution-benchmark/`), MPS, warm steps:

| setting | ms/example | parameters | device memory | voxels |
|---|---|---|---|---|
| `32^3` (batch 32) | 13.0 | 1.8M | 1.2 GB | 32,768 |
| `64^3` (batch 16) | 104.3 | 12.9M | 2.3 GB | 262,144 |

Quality (`outputs/runs/20260611-111722-resolution-comparison/`):

- Side-by-side reconstructions: `64^3` outputs are visibly smoother — the staircase voxel
  texture on cylinders, capsules, and rounded boxes largely disappears.
- The small-torus failure mode from Devlog 8 partially improves: one of two compared small
  toruses keeps a cleaner, rounder hole at `64^3`; the other is roughly unchanged.
- Honest numbers: matched-pair recon IoU is 0.877 (`32^3`, 80 epochs) vs 0.829 (`64^3`,
  30 epochs). Two effects stack against the 64 model: it trained for ~1/3 the epochs (the
  IoU curve was still climbing at epoch 30, ~88 minutes in), and finer voxels penalize
  boundary disagreement harder. Visually it still wins.

## What Failed Or Confused Me

- I expected the higher resolution to show up as higher IoU; it showed up as *lower* IoU
  and better-looking meshes. Resolution changes what the metric means — a comparison table
  needs matched training budgets and ideally a physical (mesh-space) metric, not a
  voxel-space one.
- 88 minutes for 30 epochs is the real cost lesson: the same experiment loop I ran ~10
  times a day at `32^3` becomes a twice-a-day loop at `64^3`. Cheap iteration is a feature
  of low resolution, not just low cost.

## What This Teaches About Modern 3D Generation

This is exactly why TRELLIS uses sparse voxels and EG3D uses triplanes: dense grids price
you out at `O(R^3)` while almost all of the information lives near the surface at
`O(R^2)`. The measured 8x-per-resolution-doubling table is the budget argument behind every
compact 3D representation paper.

## Next Step

Milestone 10: package the pipeline as a final demo — CLI, web gallery, architecture
write-up, and the comparison to the original SDS plan.
