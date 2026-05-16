# mini-latent

`mini-latent` is a low-level learning project for understanding modern 3D generation at a small,
affordable scale.

The long-term goal is to build a tiny native-3D pipeline:

```text
text label
  -> condition vector
  -> tiny latent generator
  -> 3D latent
  -> decoder
  -> voxel grid / TSDF
  -> mesh export
```

Instead of starting with huge models or giant datasets, the project builds each piece from first
principles.

## Current Status

Milestone 1 is complete: a procedural 3D dataset generator.

It currently creates:

- `32 x 32 x 32` occupancy grids
- spheres, cubes, rounded boxes, cylinders, capsules, and tori
- labels such as `red medium matte sphere` and `blue small wide cylinder`
- train/validation splits
- metadata, dataset statistics, and preview images

The first generated dataset contains:

```text
1000 training examples
200 validation examples
```

## Quickstart

Use the repo virtual environment:

```bash
./venv/bin/python -m tiny3dlatent.data.generate --config configs/procedural_dataset.json
```

This writes:

```text
data/procedural/
  train/
  val/
  metadata.json
  dataset_stats.json
```

To generate preview images from the existing dataset:

```bash
./venv/bin/python -m tiny3dlatent.data.preview_cli
```

To run the tests:

```bash
./venv/bin/python -m pytest tests/test_procedural_dataset.py
```

## Why Procedural Data First?

Before training a 3D model, I want a dataset I can fully inspect and explain.

Procedural shapes are useful early because they avoid several real-dataset problems at once:

- inconsistent meshes
- licensing questions
- scale normalization
- voxelization edge cases
- large downloads

Real 3D datasets are planned later, after the core pipeline works.

## Repository Layout

```text
tiny3dlatent/
  data/        procedural dataset generation and previews
  utils/       small shared helpers

configs/       dataset configuration
tests/         focused tests for dataset generation
docs/          project plan, explainers, and devlogs
```

## Documentation

- [Project plan](docs/LOW_LEVEL_MODERN_3D_PROJECT_PLAN.md)
- [Visual project roadmap](docs/low_level_modern_3d_plan_summary.html)
- [Dataset explainer](docs/dataset_explained_simple.html)
- [Devlog 1: Building a tiny 3D dataset](docs/devlog_01_procedural_3d_dataset.md)

## Next Milestone

Milestone 2 will turn voxel grids into visible mesh assets:

```text
occupancy grid
  -> marching cubes
  -> mesh
  -> preview render
```

That will make the 3D structure easier to inspect and prepare the project for later learned
representations.
