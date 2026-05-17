# Devlog 1: Building A Tiny 3D Dataset Instead Of Downloading A Giant One

## Goal

The goal of this milestone was to create the first dataset for `Tiny3D-Latent`.

I did not want to start by downloading a huge real-world 3D dataset. That would bring in many
problems at once: inconsistent mesh quality, licensing, normalization, voxelization, and large
downloads. I first wanted a small dataset that I fully understand and can generate myself.

So this milestone answers one basic question:

```text
Can I create simple 3D training examples from scratch and save them in a form a neural network can use later?
```

## What I Built

I built a procedural 3D dataset generator inside the new `tiny3dlatent` package.

The generator creates:

- spheres
- cubes
- rounded boxes
- cylinders
- capsules
- tori

Each example is saved as a `32 x 32 x 32` occupancy grid:

```text
0 = empty space
1 = inside the object
```

The generated dataset currently contains:

| Split | Count |
|---|---:|
| Train | 1000 |
| Validation | 200 |
| Total | 1200 |

The shapes are also given labels such as:

```text
yellow medium tall torus
red medium matte sphere
blue small wide cylinder
```

The generator saves:

```text
data/procedural/
  train/
  val/
  metadata.json
  dataset_stats.json
```

It also creates preview artifacts under `outputs/runs/`, including:

- `preview_grid.png`
- `example_slices.png`
- run metadata
- dataset statistics

## The Key Idea In Plain English

A normal image dataset contains many 2D images.

A voxel dataset contains many 3D grids.

For example, a `32 x 32` image is made of pixels:

```text
height x width
```

A `32 x 32 x 32` 3D object is made of voxels:

```text
depth x height x width
```

Each voxel is just a tiny cube in space. In this milestone, each voxel stores one simple answer:

```text
Is this tiny part of space inside the object or not?
```

That means the model will not need photographs to learn the first shape representation. It can
learn directly from the 3D grids.

## One Example

One generated record looks like this:

```json
{
  "id": "shape_000001",
  "label": "yellow medium tall torus",
  "shape_type": "torus",
  "grid_shape": [32, 32, 32],
  "grid_dtype": "uint8",
  "grid_file": "data/procedural/train/shape_000001_occupancy.npy"
}
```

The `.npy` file contains the actual 3D grid. Later milestones will feed grids like this into a 3D
autoencoder.

## Code References

The implementation for this milestone is small enough to read directly:

- [Dataset generator](https://github.com/antash-mishra/mini-latent/blob/main/tiny3dlatent/data/generate.py)
- [Shape formulas](https://github.com/antash-mishra/mini-latent/blob/main/tiny3dlatent/data/shapes.py)
- [3D grid creation](https://github.com/antash-mishra/mini-latent/blob/main/tiny3dlatent/data/grid.py)
- [Preview rendering](https://github.com/antash-mishra/mini-latent/blob/main/tiny3dlatent/data/preview.py)
- [Dataset config](https://github.com/antash-mishra/mini-latent/blob/main/configs/procedural_dataset.json)
- [Tests](https://github.com/antash-mishra/mini-latent/blob/main/tests/test_procedural_dataset.py)

## What I Studied

This milestone introduced a few ideas that will keep showing up later:

- coordinate grids in 3D
- occupancy grids
- implicit shape formulas
- procedural data generation
- train/validation splits
- reproducibility through random seeds

The important realization is that a dataset does not have to begin with downloaded assets. For an
early learning project, synthetic data is useful because every part of it is inspectable.

## Main Result

The dataset now contains a balanced mix of six shape families:

| Shape Type | Count |
|---|---:|
| Capsule | 198 |
| Cube | 201 |
| Cylinder | 203 |
| Rounded box | 202 |
| Sphere | 176 |
| Torus | 220 |

The colors are also distributed across the dataset:

| Color | Count |
|---|---:|
| Blue | 191 |
| Cyan | 198 |
| Green | 192 |
| Orange | 208 |
| Red | 207 |
| Yellow | 204 |

The latest generated preview artifacts are here:

```text
outputs/runs/20260517-022123-preview/
```

## What Confused Me

The preview images were useful, but they also taught an important lesson.

At first glance, some torus examples did not look like donuts. A `green medium wide torus`, for
example, looked like two separate green blobs.

That happened because the preview currently shows only one flat middle slice through the 3D grid.
For a rotated torus, one slice often cuts through only two small parts of the ring:

```text
3D torus:
  a full donut shape

single 2D slice:
  two separated pieces
```

So the object was still a torus in 3D. The preview was simply not showing enough of the volume to
make that obvious.

This is a useful warning for the rest of the project:

```text
a weak visualization can make correct 3D data look wrong
```

For later improvements, the preview grid should probably use a max projection or a proper mesh
render instead of only a single slice.

## What This Teaches About Modern 3D Generation

Modern systems use much richer 3D representations than this tiny dataset, but the same broad
questions already appear here:

- How do we represent 3D structure?
- How do we save it compactly?
- How do we inspect whether the data is correct?
- How do we make the data diverse enough for a model to learn from?

The large models come later. First, the representation has to make sense.

## Current Limitations

This dataset is intentionally simple.

- It does not contain real objects like cups or chairs yet.
- `metallic` and `matte` are currently labels only; they do not change the voxel geometry or add
  material channels yet.
- The preview grid is only a 2D slice, not a true 3D render.
- The dataset is synthetic, so it is cleaner than real-world mesh datasets.

Those limitations are fine for now. The job of this milestone was not realism. It was to build a
clean first training dataset that the next model can learn from.

## Next Step

The next milestone is to turn the voxel grids into actual mesh assets:

```text
occupancy grid -> marching cubes -> mesh -> preview render
```

That will make the 3D nature of the data much easier to inspect and will prepare the project for
later generated meshes.
