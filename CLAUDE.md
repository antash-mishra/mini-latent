# CLAUDE.md

`mini-latent` (Python package: `tiny3dlatent`) is a low-level learning project that builds a
tiny native-3D generative pipeline from scratch: procedural voxel data -> 3D VAE -> latent
flow generator -> marching cubes mesh export, with simple text conditioning and materials.

## Project Plan

The single source of truth for milestones, scope, and verification is
[docs/LOW_LEVEL_MODERN_3D_PROJECT_PLAN.md](docs/LOW_LEVEL_MODERN_3D_PROJECT_PLAN.md).

- The plan's "Current Status" section records which milestones are done.
- Every milestone has a `Verification:` block describing how to prove it is complete.
  Follow it before declaring a milestone done.

## Code Structure

```text
tiny3dlatent/          Python package (the actual pipeline)
  data/                procedural dataset generation, labels, previews (Milestone 1)
  representation/      marching cubes, mesh cleanup, OBJ export, stats, previews (Milestone 2)
  models/              torch dataset, AE/VAE, latent flows, training + generation CLIs (Milestones 3-6)
  text/                tiny-vocabulary prompt parser and attribute embedding indices (Milestone 6)
  utils/               small shared helpers (io, rng)
configs/               JSON configs (e.g. procedural_dataset.json)
tests/                 pytest tests
docs/                  project plan, devlogs, explainers, milestone notes
data/procedural/       generated dataset (train/, val/, metadata.json, dataset_stats.json)
outputs/runs/          timestamped run folders with config, metadata, stats, previews
src/components/        TypeScript interactive blog explainers (NOT part of the pipeline)
venv/                  repo virtual environment — always use this, not system Python
```

Future milestones add packages under `tiny3dlatent/` (e.g. `models/` for the
autoencoder/VAE/generator) — see the plan.

## Commands

Always use the repo venv interpreter `./venv/bin/python`.

```bash
# Generate the procedural dataset (writes data/procedural/ and a run folder)
./venv/bin/python -m tiny3dlatent.data.generate --config configs/procedural_dataset.json

# Preview images from the existing dataset
./venv/bin/python -m tiny3dlatent.data.preview_cli

# Extract meshes from dataset grids (OBJ + stats + renders into a run folder)
./venv/bin/python -m tiny3dlatent.representation.extract_cli --config configs/mesh_extraction.json

# Train the 3D autoencoder (checkpoint + history + recon grid into a run folder)
./venv/bin/python -m tiny3dlatent.models.train_ae --config configs/autoencoder.json
# Overfit-8 sanity check before a full run
./venv/bin/python -m tiny3dlatent.models.train_ae --overfit 8 --epochs 200

# Train the 3D VAE, then render latent interpolation strips from its checkpoint
./venv/bin/python -m tiny3dlatent.models.train_vae --config configs/vae.json
./venv/bin/python -m tiny3dlatent.models.interpolate_cli --steps 8

# Train the class-conditioned latent flow on the latest VAE, then generate + mesh samples
./venv/bin/python -m tiny3dlatent.models.train_flow --config configs/latent_flow.json
./venv/bin/python -m tiny3dlatent.models.generate_cli --per-class 16 --steps 50

# Train the prompt-conditioned flow (CFG), then generate from text prompts
./venv/bin/python -m tiny3dlatent.models.train_text_flow --config configs/text_flow.json
./venv/bin/python -m tiny3dlatent.models.text_generate_cli --prompt "blue tall cylinder"

# Color/material: train the color VAE, retrain the flow on it, export GLB assets
./venv/bin/python -m tiny3dlatent.models.train_color_vae --config configs/color_vae.json
./venv/bin/python -m tiny3dlatent.models.train_text_flow --config configs/color_text_flow.json
./venv/bin/python -m tiny3dlatent.models.asset_generate_cli --prompt "red metallic sphere"

# Build the HTML generation report end-to-end (recon + generation + failures)
./venv/bin/python -m tiny3dlatent.report.report_cli

# Tests
./venv/bin/python -m pytest tests/

# Lint / format (config in pyproject.toml: py312, line-length 88)
./venv/bin/ruff check .
./venv/bin/ruff format .
```

## How To Verify A Milestone Is Done

Each milestone's `Verification:` block in the plan is authoritative. The general policy:

1. Programmatic checks first: tests, metrics with thresholds, trimesh assertions
   (watertightness, bounds, vertex/face counts), stats files.
2. Visual checks second: generated preview images and renders are saved into
   `outputs/runs/<timestamp>-<experiment>/`. Read the image files directly, or open HTML
   reports in a browser and take a screenshot.
3. Never declare a milestone done from code review alone — run the artifact command and
   look at the output.

Quick reference (details in the plan):

| Milestone | Primary verification method |
|---|---|
| 0 Skeleton | File checks: run folder with metadata/config/stats |
| 1 Dataset | pytest + stats counts + screenshot of `preview_grid.png` |
| 2 Mesh extraction | trimesh watertight/bounds asserts + screenshot of mesh renders |
| 3 Autoencoder | overfit-8 sanity, val IoU > ~0.85, screenshot of recon grid |
| 4 VAE | IoU vs AE, KL-not-collapsed check, screenshot of interpolation strip |
| 5 Latent generator | non-empty/connected/voxel-count checks, screenshot of per-class grid |
| 6 Text conditioning | aspect-ratio checks for tall/wide, prompt metadata, screenshot of prompt gallery |
| 7 Color/material | GLB material asserts vs `COLOR_RGB` palette, screenshot of colored renders |
| 8 Metrics/report | report script runs end-to-end, screenshot of `generation_report.html` in browser |
| 9 Resolution | measured memory/time table, screenshot of 32^3 vs 64^3 comparison |
| 10 Final demo | clean-environment end-to-end CLI run, screenshot of web gallery |
| 11 Real data | license table complete, normalization asserts, screenshot of `real_dataset_report.html` |

## Conventions

- Every experiment writes a timestamped folder under `outputs/runs/` containing
  `config.json`, `metadata.json`, and its artifacts. Do not write outputs elsewhere.
- Configs are plain JSON in `configs/`; code merges them over an in-module `DEFAULT_CONFIG`.
- Datasets store one `*_occupancy.npy` (`uint8`, values 0/1, shape `R^3`) per example, with
  all labels and paths in `metadata.json`.
- Grids use normalized `[-1, 1]` coordinates with `indexing="ij"` (see
  `tiny3dlatent/data/grid.py`); keep new representation code consistent with this.
- After each milestone, write a devlog in `docs/` (`devlog_NN_*.md`) using the template at
  the bottom of the project plan.
- Keep it cheap: tiny networks, `32^3` first, CPU/MPS-friendly, no paid APIs, no large
  dataset downloads before Milestone 11.
