from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from tiny3dlatent.data.grid import make_grid
from tiny3dlatent.data.labels import SHAPE_TYPES
from tiny3dlatent.data.shapes import ShapeSpec, generate_occupancy
from tiny3dlatent.representation.cleanup import clean_mesh
from tiny3dlatent.representation.export import export_mesh
from tiny3dlatent.representation.marching_cubes import extract_mesh_from_occupancy
from tiny3dlatent.representation.stats import mesh_stats
from tiny3dlatent.utils.io import write_json


def make_test_occupancy(shape_type: str = "sphere", resolution: int = 32) -> np.ndarray:
    spec = ShapeSpec(
        shape_type=shape_type,  # type: ignore[arg-type]
        color="red",
        size="medium",
        descriptor="standard",
        center=(0.0, 0.0, 0.0),
        scale=(0.55, 0.55, 0.55),
        rotation=(0.0, 0.0, 0.0),
    )
    return generate_occupancy(spec, make_grid(resolution))


def test_marching_cubes_produces_valid_trimesh() -> None:
    mesh = extract_mesh_from_occupancy(make_test_occupancy())

    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_mesh_vertices_stay_in_normalized_space() -> None:
    mesh = extract_mesh_from_occupancy(make_test_occupancy())

    assert mesh.bounds.min() >= -1.1
    assert mesh.bounds.max() <= 1.1


def test_padding_prevents_boundary_holes() -> None:
    occupancy = np.zeros((8, 8, 8), dtype=np.uint8)
    occupancy[0:5, 2:6, 2:6] = 1

    mesh = extract_mesh_from_occupancy(occupancy)

    assert mesh.is_watertight


def test_cleanup_preserves_non_empty_mesh() -> None:
    raw = extract_mesh_from_occupancy(make_test_occupancy())
    cleaned = clean_mesh(raw)

    assert len(cleaned.faces) > 0
    assert len(cleaned.vertices) <= len(raw.vertices)


def test_all_shape_types_extract_watertight_single_component() -> None:
    for shape_type in SHAPE_TYPES:
        mesh = clean_mesh(extract_mesh_from_occupancy(make_test_occupancy(shape_type)))
        stats = mesh_stats(mesh)

        assert stats["vertex_count"] > 0, shape_type
        assert stats["face_count"] > 0, shape_type
        assert stats["is_watertight"], shape_type
        assert stats["connected_components"] == 1, shape_type


def test_sphere_mesh_matches_analytic_expectations() -> None:
    mesh = clean_mesh(extract_mesh_from_occupancy(make_test_occupancy("sphere")))
    stats = mesh_stats(mesh)

    radius = 0.55
    analytic_volume = 4.0 / 3.0 * np.pi * radius**3
    assert stats["volume"] == pytest.approx(analytic_volume, rel=0.25)

    bounds = np.abs(mesh.bounds)
    assert bounds.max() <= radius + 0.1


def test_rejects_invalid_grids() -> None:
    with pytest.raises(ValueError):
        extract_mesh_from_occupancy(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        extract_mesh_from_occupancy(np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        extract_mesh_from_occupancy(np.zeros((4, 4, 8), dtype=np.uint8))


def test_export_and_stats_roundtrip(tmp_path: Path) -> None:
    mesh = clean_mesh(extract_mesh_from_occupancy(make_test_occupancy()))
    obj_path = export_mesh(mesh, tmp_path / "meshes" / "sphere.obj")

    assert obj_path.exists()
    reloaded = trimesh.load(obj_path, force="mesh")
    assert len(reloaded.faces) == len(mesh.faces)

    stats = mesh_stats(mesh, obj_path)
    assert stats["file_size_bytes"] > 0
    write_json(tmp_path / "mesh_stats.json", stats)
    assert json.loads((tmp_path / "mesh_stats.json").read_text())["is_watertight"]
