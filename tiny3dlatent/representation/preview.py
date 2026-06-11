from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tiny3dlatent.data.labels import COLOR_RGB
from tiny3dlatent.utils.io import ensure_dir


def save_mesh_preview_grid(
    entries: list[tuple[trimesh.Trimesh, dict[str, object]]],
    output_path: Path,
    *,
    columns: int = 4,
) -> None:
    """Render a grid of shaded meshes, one panel per entry."""
    if not entries:
        raise ValueError("at least one mesh is required for preview")

    ensure_dir(output_path.parent)
    rows = int(np.ceil(len(entries) / columns))
    figure = plt.figure(figsize=(columns * 2.6, rows * 2.8))

    for index, (mesh, metadata) in enumerate(entries):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        _draw_mesh(
            axis,
            mesh,
            color=str(metadata.get("color", "blue")),
            rgb=metadata.get("rgb"),
        )
        axis.set_title(str(metadata.get("label", "")), fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_mesh_views(
    mesh: trimesh.Trimesh,
    output_path: Path,
    *,
    title: str,
    color: str,
) -> None:
    """Render front/side/top views of a single mesh."""
    ensure_dir(output_path.parent)
    views = [("front", 0, -90), ("side", 0, 0), ("top", 90, -90)]
    figure = plt.figure(figsize=(8.4, 3.0))

    for index, (name, elev, azim) in enumerate(views):
        axis = figure.add_subplot(1, 3, index + 1, projection="3d")
        _draw_mesh(axis, mesh, color=color, elev=elev, azim=azim)
        axis.set_title(name, fontsize=9)

    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _draw_mesh(
    axis: plt.Axes,
    mesh: trimesh.Trimesh,
    *,
    color: str,
    rgb: object = None,
    elev: float = 22.0,
    azim: float = -60.0,
) -> None:
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.float32)
    else:
        rgb = (
            np.array(COLOR_RGB.get(color, COLOR_RGB["blue"]), dtype=np.float32) / 255.0
        )
    triangles = mesh.vertices[mesh.faces]
    collection = Poly3DCollection(
        triangles,
        shade=True,
        facecolors=np.tile(rgb, (len(triangles), 1)),
    )
    axis.add_collection3d(collection)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_zlim(-1, 1)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elev, azim=azim)
    axis.axis("off")
