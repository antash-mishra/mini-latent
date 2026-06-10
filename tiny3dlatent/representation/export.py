from __future__ import annotations

from pathlib import Path

import trimesh

from tiny3dlatent.utils.io import ensure_dir


def export_mesh(mesh: trimesh.Trimesh, path: Path) -> Path:
    """Export a mesh to disk; format is inferred from the file extension."""
    ensure_dir(path.parent)
    mesh.export(path)
    return path
