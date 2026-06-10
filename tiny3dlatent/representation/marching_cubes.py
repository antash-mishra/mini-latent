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
    """Convert a uint8 occupancy grid into a world-space triangle mesh.

    The grid is padded with empty voxels so shapes touching the border still
    produce closed surfaces, then vertices are mapped back into the same
    normalized [-1, 1] space used by `tiny3dlatent.data.grid.make_grid`.
    """
    if occupancy.ndim != 3:
        raise ValueError("occupancy must be a 3D array")
    if min(occupancy.shape) < 2:
        raise ValueError("occupancy grid must be at least 2 voxels along every axis")

    resolution = occupancy.shape[0]
    if occupancy.shape != (resolution, resolution, resolution):
        raise ValueError("expected a cubic occupancy grid")
    if occupancy.max() == occupancy.min():
        raise ValueError("occupancy grid has no inside/outside boundary to extract")

    scalar_field = occupancy.astype(np.float32, copy=False)
    padded = np.pad(
        scalar_field,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )

    vertices, faces, normals, _ = marching_cubes(padded, level=iso_value)

    voxel_size = 2.0 / (resolution - 1)
    vertices = -1.0 + (vertices - pad_width) * voxel_size

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
