# Marching Cubes Implementation Notes For Milestone 2

## Goal

Implement a small, reliable pipeline that converts the occupancy grids already produced in
Milestone 1 into clean mesh assets:

```text
uint8 occupancy grid
  -> padded scalar field
  -> marching cubes
  -> world-space vertices
  -> cleaned trimesh
  -> OBJ + mesh stats + previews
```

This document maps directly onto the planned Milestone 2 package structure:

```text
tiny3dlatent/
  representation/
    marching_cubes.py
    cleanup.py
    export.py
    stats.py
    preview.py
```

## What we already know from the current codebase

The existing Milestone 1 code gives us the exact contract:

- `tiny3dlatent/data/grid.py`
  - builds normalized coordinates in `[-1, 1]`
  - uses `indexing="ij"`
- `tiny3dlatent/data/shapes.py`
  - returns occupancy grids as `np.uint8`
  - grids contain only `0` and `1`
- `configs/procedural_dataset.json`
  - currently uses resolution `32`

So the first extractor only needs to support:

```python
occupancy.shape == (resolution, resolution, resolution)
occupancy.dtype == np.uint8
occupancy.min() == 0
occupancy.max() == 1
```

## Recommended implementation order

### 1. Start with the extractor only

Do not begin with previews, CLI plumbing, or exports.

First prove:

```text
known occupancy grid -> valid trimesh
```

### 2. Then add cleanup

Only after extraction works:

```text
raw mesh -> cleaned mesh
```

### 3. Then add export and stats

Once meshes are trustworthy:

```text
cleaned mesh -> mesh.obj + mesh_stats.json
```

### 4. Finish with previews and batch CLI

The turntable and experiment runner should be last, because they are consumers of the geometry path,
not prerequisites for it.

## Core extraction example

This is the most important function Milestone 2 needs:

```python
from __future__ import annotations

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def extract_mesh_from_occupancy(
    occupancy: np.ndarray,
    *,
    iso_value: float = 0.5,
    pad_width: int = 1,
) -> trimesh.Trimesh:
    """Convert a uint8 occupancy grid into a world-space triangle mesh."""
    if occupancy.ndim != 3:
        raise ValueError("occupancy must be a 3D array")
    if min(occupancy.shape) < 2:
        raise ValueError("occupancy grid must be at least 2 voxels along every axis")
    if occupancy.max() == occupancy.min():
        raise ValueError("occupancy grid has no inside/outside boundary to extract")

    resolution = occupancy.shape[0]
    if occupancy.shape != (resolution, resolution, resolution):
        raise ValueError("expected a cubic occupancy grid")

    scalar_field = occupancy.astype(np.float32, copy=False)
    padded = np.pad(
        scalar_field,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )

    vertices, faces, normals, _ = marching_cubes(
        padded,
        level=iso_value,
    )

    voxel_size = 2.0 / (resolution - 1)
    vertices = -1.0 + (vertices - pad_width) * voxel_size

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
```

### Why each piece exists

| Code | Reason |
|---|---|
| `occupancy.astype(np.float32)` | marching cubes expects scalar values, not just semantic labels |
| `np.pad(..., constant_values=0.0)` | guarantees empty space outside the original grid |
| `level=0.5` | extracts the surface halfway between `0` and `1` |
| `vertices = -1 + (vertices - pad_width) * voxel_size` | converts padded index-space vertices back to normalized world space |
| `process=False` | keeps extraction and cleanup as separate, testable steps |

## Minimal cleanup example

```python
import trimesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply basic mesh hygiene after marching cubes."""
    cleaned = mesh.copy()
    cleaned.merge_vertices()
    cleaned.remove_degenerate_faces()
    cleaned.remove_unreferenced_vertices()
    cleaned.fix_normals()
    return cleaned
```

If the installed `trimesh` version no longer exposes `remove_degenerate_faces()`, the equivalent
fallback is:

```python
cleaned.update_faces(cleaned.nondegenerate_faces())
```

## Tiny end-to-end example using the existing project code

This example creates a centered sphere using the same functions already used in Milestone 1:

```python
from tiny3dlatent.data.grid import make_grid
from tiny3dlatent.data.shapes import ShapeSpec, generate_occupancy


grid = make_grid(32)
spec = ShapeSpec(
    shape_type="sphere",
    color="red",
    size="medium",
    descriptor="standard",
    center=(0.0, 0.0, 0.0),
    scale=(0.55, 0.55, 0.55),
    rotation=(0.0, 0.0, 0.0),
)

occupancy = generate_occupancy(spec, grid)
mesh = extract_mesh_from_occupancy(occupancy)
mesh = clean_mesh(mesh)

print(mesh.vertices.shape)
print(mesh.faces.shape)
print(mesh.bounds)
print(mesh.is_watertight)
```

What you should expect:

- non-zero vertices and faces
- bounds roughly inside `[-1, 1]`
- a watertight mesh for the centered sphere

## Export example

```python
from pathlib import Path
import trimesh


def export_obj(mesh: trimesh.Trimesh, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path
```

For the first version, geometry-only OBJ export is enough. Materials can wait until the later color
milestone.

## Mesh statistics example

```python
from pathlib import Path
import trimesh


def mesh_stats(mesh: trimesh.Trimesh, obj_path: Path | None = None) -> dict[str, object]:
    stats: dict[str, object] = {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bounds_min": mesh.bounds[0].tolist(),
        "bounds_max": mesh.bounds[1].tolist(),
        "volume": float(mesh.volume),
        "surface_area": float(mesh.area),
        "is_watertight": bool(mesh.is_watertight),
    }
    if obj_path is not None and obj_path.exists():
        stats["file_size_bytes"] = int(obj_path.stat().st_size)
    return stats
```

## Tests worth writing first

### 1. A centered sphere produces a mesh

```python
def test_marching_cubes_produces_valid_trimesh() -> None:
    occupancy = make_test_sphere_occupancy()
    mesh = extract_mesh_from_occupancy(occupancy)

    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
```

### 2. Vertices stay in normalized coordinates

```python
def test_mesh_vertices_stay_in_normalized_space() -> None:
    occupancy = make_test_sphere_occupancy()
    mesh = extract_mesh_from_occupancy(occupancy)

    assert mesh.bounds.min() >= -1.1
    assert mesh.bounds.max() <= 1.1
```

The small tolerance is useful because the extracted surface lies halfway between sampled voxels.

### 3. Padding closes shapes that touch the border

```python
def test_padding_prevents_boundary_holes() -> None:
    occupancy = np.zeros((8, 8, 8), dtype=np.uint8)
    occupancy[0:5, 2:6, 2:6] = 1

    mesh = extract_mesh_from_occupancy(occupancy)

    assert mesh.is_watertight
```

### 4. Cleanup never destroys a valid mesh

```python
def test_cleanup_preserves_non_empty_mesh() -> None:
    occupancy = make_test_sphere_occupancy()
    raw = extract_mesh_from_occupancy(occupancy)
    cleaned = clean_mesh(raw)

    assert len(cleaned.faces) > 0
    assert len(cleaned.vertices) <= len(raw.vertices)
```

## Common mistakes to avoid

### Mistake 1: forgetting coordinate conversion

Symptom:

```text
mesh bounds look like [0, 31] instead of around [-1, 1]
```

Fix:

```python
vertices = -1.0 + (vertices - pad_width) * voxel_size
```

### Mistake 2: padding but not compensating for the padding

Symptom:

```text
mesh is shifted by one voxel
```

Fix:

```python
vertices - pad_width
```

before scaling back into world space.

### Mistake 3: testing only spheres

Spheres are forgiving. Also test:

- cubes for sharp edges
- tori for holes
- capsules for elongated geometry
- a boundary-touching object for padding behavior

### Mistake 4: treating `32^3` as final quality

`32^3` is a development resolution. Its job is to prove the pipeline cheaply, not to produce the
best-looking asset.

### Mistake 5: mixing representation experiments with extraction bugs

Do not add TSDF, textures, smoothing, and CLI complexity before the occupancy path is boringly
correct.

## How this evolves later

The extractor interface should remain stable:

```python
grid-like field -> mesh
```

Only the field changes over time:

| Project phase | Field sent into marching cubes |
|---|---|
| Milestone 2 | procedural occupancy grid |
| Later geometry milestone | predicted occupancy grid |
| Optional TSDF experiment | signed-distance field |
| Future higher-quality version | denser or learned field |

That is why it is worth making this module small, explicit, and well-tested now.

## Suggested Milestone 2 reading order

1. Read `docs/marching_cubes_explained.md`
2. Re-read `tiny3dlatent/data/grid.py`
3. Re-read `tiny3dlatent/data/shapes.py`
4. Implement `representation/marching_cubes.py`
5. Add tests before preview/render code
6. Only then build the experiment runner

