from __future__ import annotations

from pathlib import Path

import trimesh

from tiny3dlatent.utils.io import ensure_dir


def export_mesh(mesh: trimesh.Trimesh, path: Path) -> Path:
    """Export a mesh to disk; format is inferred from the file extension."""
    ensure_dir(path.parent)
    mesh.export(path)
    return path


def export_glb_with_material(
    mesh: trimesh.Trimesh,
    path: Path,
    *,
    base_color: tuple[float, float, float],
    roughness: float,
    metallic: float,
    name: str = "material",
) -> Path:
    """Export a GLB with a uniform PBR material (base color in [0, 1])."""
    ensure_dir(path.parent)
    material = trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=[*base_color, 1.0],
        roughnessFactor=float(roughness),
        metallicFactor=float(metallic),
    )
    export = mesh.copy()
    export.visual = trimesh.visual.TextureVisuals(material=material)
    export.export(path)
    return path


def load_glb_material(path: Path) -> dict[str, object]:
    """Load a GLB and return its first mesh's PBR material parameters."""
    scene = trimesh.load(path, force="scene")
    meshes = list(scene.geometry.values())
    if not meshes:
        raise ValueError(f"no geometry found in {path}")
    material = meshes[0].visual.material
    base_color = [float(v) for v in material.baseColorFactor[:3]]
    if any(value > 1.0 for value in base_color):
        base_color = [value / 255.0 for value in base_color]
    return {
        "base_color": base_color,
        "roughness": float(material.roughnessFactor),
        "metallic": float(material.metallicFactor),
        "name": material.name,
    }
