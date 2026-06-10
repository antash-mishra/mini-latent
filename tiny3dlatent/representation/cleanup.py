from __future__ import annotations

import trimesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply basic mesh hygiene after marching cubes."""
    cleaned = mesh.copy()
    cleaned.merge_vertices()
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()
    cleaned.fix_normals()
    return cleaned
