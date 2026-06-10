from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tiny3dlatent.data.generate import generate_dataset
from tiny3dlatent.data.grid import make_grid
from tiny3dlatent.data.labels import SHAPE_TYPES
from tiny3dlatent.data.shapes import ShapeSpec, generate_occupancy


def test_make_grid_uses_ij_indexing_and_normalized_bounds() -> None:
    x, y, z = make_grid(4)

    assert x.shape == (4, 4, 4)
    assert y.shape == (4, 4, 4)
    assert z.shape == (4, 4, 4)
    assert np.isclose(x[0, 0, 0], -1.0)
    assert np.isclose(x[-1, 0, 0], 1.0)
    assert np.isclose(y[0, -1, 0], 1.0)
    assert np.isclose(z[0, 0, -1], 1.0)


def test_shape_generators_return_non_empty_uint8_grids() -> None:
    grid = make_grid(32)

    for shape_type in SHAPE_TYPES:
        spec = ShapeSpec(
            shape_type=shape_type,  # type: ignore[arg-type]
            color="red",
            size="medium",
            descriptor="standard",
            center=(0.0, 0.0, 0.0),
            scale=(0.55, 0.55, 0.55),
            rotation=(0.0, 0.0, 0.0),
        )

        occupancy = generate_occupancy(spec, grid)

        assert occupancy.shape == (32, 32, 32)
        assert occupancy.dtype == np.uint8
        assert occupancy.min() == 0
        assert occupancy.max() == 1
        assert 0 < occupancy.sum() < occupancy.size


def test_generate_dataset_writes_expected_files(tmp_path: Path) -> None:
    result = generate_dataset(
        {
            "seed": 7,
            "resolution": 16,
            "train_count": 6,
            "val_count": 3,
            "dataset_dir": (tmp_path / "procedural").as_posix(),
            "output_dir": (tmp_path / "runs").as_posix(),
            "preview_count": 4,
        }
    )

    dataset_dir = Path(result["dataset_dir"])
    run_dir = Path(result["run_dir"])

    assert len(list((dataset_dir / "train").glob("*.npy"))) == 6
    assert len(list((dataset_dir / "val").glob("*.npy"))) == 3
    assert (dataset_dir / "metadata.json").exists()
    assert (dataset_dir / "dataset_stats.json").exists()
    assert (run_dir / "preview_grid.png").exists()
    assert (run_dir / "example_slices.png").exists()

    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    assert metadata["total_count"] == 9
    assert len(metadata["records"]) == 9

    first_record = metadata["records"][0]
    first_grid = np.load(first_record["grid_file"])
    assert first_grid.shape == (16, 16, 16)
    assert first_grid.dtype == np.uint8


def test_generation_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    config = {
        "seed": 12,
        "resolution": 16,
        "train_count": 3,
        "val_count": 2,
        "preview_count": 2,
    }
    first = generate_dataset(
        {
            **config,
            "dataset_dir": (tmp_path / "first" / "procedural").as_posix(),
            "output_dir": (tmp_path / "first" / "runs").as_posix(),
        }
    )
    second = generate_dataset(
        {
            **config,
            "dataset_dir": (tmp_path / "second" / "procedural").as_posix(),
            "output_dir": (tmp_path / "second" / "runs").as_posix(),
        }
    )

    first_metadata = json.loads(
        (Path(first["dataset_dir"]) / "metadata.json").read_text()
    )
    second_metadata = json.loads(
        (Path(second["dataset_dir"]) / "metadata.json").read_text()
    )

    for first_record, second_record in zip(
        first_metadata["records"], second_metadata["records"], strict=True
    ):
        assert first_record["label"] == second_record["label"]
        assert first_record["shape_type"] == second_record["shape_type"]
        assert first_record["parameters"] == second_record["parameters"]
        np.testing.assert_array_equal(
            np.load(first_record["grid_file"]),
            np.load(second_record["grid_file"]),
        )
