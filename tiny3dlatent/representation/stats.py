from __future__ import annotations

from pathlib import Path

import trimesh


def mesh_stats(
    mesh: trimesh.Trimesh, mesh_path: Path | None = None
) -> dict[str, object]:
    """Summarize a mesh for run metadata and milestone verification."""
    stats: dict[str, object] = {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "bounds_min": mesh.bounds[0].tolist(),
        "bounds_max": mesh.bounds[1].tolist(),
        "surface_area": float(mesh.area),
        "is_watertight": bool(mesh.is_watertight),
        "connected_components": len(mesh.split(only_watertight=False)),
    }
    if mesh.is_watertight:
        stats["volume"] = float(mesh.volume)
    if mesh_path is not None and mesh_path.exists():
        stats["file_size_bytes"] = int(mesh_path.stat().st_size)
    return stats
