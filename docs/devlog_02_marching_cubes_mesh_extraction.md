# Devlog 2: Turning a voxel grid into my first generated mesh

## Goal

Prove that the occupancy grids from Milestone 1 can become real, inspectable mesh assets:

```text
uint8 occupancy grid
  -> padded scalar field
  -> marching cubes
  -> world-space vertices
  -> cleaned trimesh
  -> OBJ + mesh stats + previews
```

If this path works on procedural grids today, it will work unchanged on *predicted* grids
once the autoencoder and generator exist. The decoder output changes; the asset path does not.

## What I Built

A new `tiny3dlatent/representation/` package:

- `marching_cubes.py` — `extract_mesh_from_occupancy()`: validates the grid, pads it with one
  layer of empty voxels, runs scikit-image marching cubes at iso level 0.5, and maps vertices
  back into the same normalized `[-1, 1]` space that `data/grid.py` uses.
- `cleanup.py` — `clean_mesh()`: merge vertices, drop degenerate faces, remove unreferenced
  vertices, fix normals.
- `export.py` — `export_mesh()`: extension-driven export (OBJ for now).
- `stats.py` — `mesh_stats()`: vertices, faces, bounds, surface area, watertightness,
  connected components, volume (watertight only), file size.
- `preview.py` — shaded matplotlib renders: a mesh grid and front/side/top views.
- `extract_cli.py` — batch runner driven by `configs/mesh_extraction.json`: picks N examples
  per shape type from the dataset, extracts/cleans/exports each, and writes
  `mesh_stats.json`, `metadata.json`, and preview images into a timestamped run folder.

Plus `tests/test_mesh_extraction.py` with 8 tests, including the four from the implementation
notes and an analytic check that the extracted sphere's volume is close to `4/3 * pi * r^3`.

New dependencies: `trimesh`, `scikit-image`, `scipy`.

## What I Studied

- Marching cubes: how the 15 canonical cube cases turn a scalar field crossing into triangles
  (`docs/marching_cubes_explained.md`, the scikit-image docs, and the Wikipedia overview).
- Why the surface lands halfway between an occupied and an empty voxel at iso level 0.5.
- trimesh's mesh hygiene model: `process=False` on construction so extraction and cleanup
  stay separate, testable steps.
- Index space vs world space: marching cubes returns vertices in padded voxel indices, so the
  conversion is `world = -1 + (index - pad_width) * (2 / (resolution - 1))`.

## Key Idea In Plain English

A voxel grid says "inside" or "outside" at sample points; a mesh is the skin between them.
Marching cubes walks every little cube of 8 neighboring samples, and wherever some corners are
inside and some are outside, it stitches triangles across the crossing. Padding the grid with
empty voxels first guarantees that even a shape pressed against the grid border gets a closed
skin, because the algorithm always sees "outside" beyond the edge.

## Main Result

From the run on the existing validation split (2 examples per shape type):

```text
extracted 12 meshes (12 watertight) -> outputs/runs/20260610-125156-mesh-extraction
```

- All 12 meshes are watertight, single-component, and inside the normalized bounds.
- `mesh_preview_grid.png` shows recognizable cylinders, capsules, cubes, rounded boxes,
  spheres, and tori — the tori have proper holes, the capsules proper caps.
- A centered radius-0.55 sphere extracts with volume within a few percent of the analytic
  value at `32^3`.
- Typical mesh size at `32^3`: roughly 300-2,000 vertices, 600-4,000 faces, 20-125 KB OBJ.

## What Failed Or Confused Me

- trimesh 4.x removed `remove_degenerate_faces()`; the fallback from the implementation notes
  (`update_faces(mesh.nondegenerate_faces())`) is now the primary path.
- matplotlib's `Poly3DCollection(shade=True, edgecolors="none")` crashes with a broadcasting
  error, because shading tries to shade the (empty) edge color array too. The fix was to pass
  per-face facecolors and let matplotlib derive edges.
- The surfaces are visibly faceted and slightly "staircased". That is not a bug — it is what
  binary occupancy at `32^3` looks like. A TSDF field would give marching cubes real gradient
  information to interpolate smoother surfaces; that is a later, optional experiment.

## What This Teaches About Modern 3D Generation

Every modern native-3D system ends with this same bridge: a learned volumetric or implicit
representation has to become an exportable asset. TRELLIS, Hunyuan3D, and TripoSG all decode
to fields and then extract meshes. Building the extraction path early means every future
milestone — autoencoder reconstructions, VAE interpolations, generated latents — gets
inspectable `.obj` files and renders for free.

## Next Step

Milestone 3: a tiny 3D convolutional autoencoder. Encode a `32^3` occupancy grid into a small
latent vector, decode it back, measure voxel IoU — and run every reconstruction through this
milestone's mesh path to see what the network actually learned.
