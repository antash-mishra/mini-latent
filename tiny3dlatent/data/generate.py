from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tiny3dlatent.data.grid import make_grid
from tiny3dlatent.data.preview import save_example_slices, save_preview_grid
from tiny3dlatent.data.shapes import generate_occupancy, sample_shape_spec
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json
from tiny3dlatent.utils.random import make_rng

DEFAULT_CONFIG = {
    "seed": 0,
    "resolution": 32,
    "train_count": 1000,
    "val_count": 200,
    "dataset_dir": "data/procedural",
    "output_dir": "outputs/runs",
    "preview_count": 20,
}


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    generate_dataset(config)


def generate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    resolution = int(config["resolution"])
    train_count = int(config["train_count"])
    val_count = int(config["val_count"])
    preview_count = int(config["preview_count"])
    dataset_dir = Path(str(config["dataset_dir"]))
    output_root = Path(str(config["output_dir"]))

    rng = make_rng(seed)
    grid = make_grid(resolution)

    run_dir = _make_run_dir(output_root)
    train_dir = ensure_dir(dataset_dir / "train")
    val_dir = ensure_dir(dataset_dir / "val")

    records: list[dict[str, Any]] = []
    preview_examples: list[tuple[np.ndarray, dict[str, object]]] = []
    first_example: tuple[np.ndarray, dict[str, object]] | None = None

    for split, count, split_dir, start_index in [
        ("train", train_count, train_dir, 1),
        ("val", val_count, val_dir, train_count + 1),
    ]:
        for offset in range(count):
            example_index = start_index + offset
            spec = sample_shape_spec(rng)
            occupancy = generate_occupancy(spec, grid)
            example_id = f"shape_{example_index:06d}"
            grid_path = split_dir / f"{example_id}_occupancy.npy"
            np.save(grid_path, occupancy)

            metadata = {
                "id": example_id,
                "split": split,
                "resolution": resolution,
                "grid_shape": list(occupancy.shape),
                "grid_dtype": str(occupancy.dtype),
                "grid_file": grid_path.as_posix(),
                "filled_voxels": int(occupancy.sum()),
                "fill_ratio": float(occupancy.mean()),
                **spec.to_metadata(),
            }
            records.append(metadata)

            if len(preview_examples) < preview_count:
                preview_examples.append((occupancy, metadata))
            if first_example is None:
                first_example = (occupancy, metadata)

    dataset_metadata = {
        "seed": seed,
        "resolution": resolution,
        "train_count": train_count,
        "val_count": val_count,
        "total_count": train_count + val_count,
        "records": records,
    }
    stats = _build_stats(records)
    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "dataset_dir": dataset_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "stats": stats,
    }

    write_json(dataset_dir / "metadata.json", dataset_metadata)
    write_json(dataset_dir / "dataset_stats.json", stats)
    write_json(run_dir / "metadata.json", run_metadata)
    write_json(run_dir / "dataset_stats.json", stats)
    write_json(run_dir / "config.json", config)

    if preview_examples:
        save_preview_grid(preview_examples, run_dir / "preview_grid.png")
    if first_example is not None:
        first_grid, first_metadata = first_example
        save_example_slices(
            first_grid,
            run_dir / "example_slices.png",
            title=str(first_metadata["label"]),
            color=str(first_metadata["color"]),
        )

    return {
        "dataset_dir": dataset_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "stats": stats,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a procedural 3D dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/procedural_dataset.json"),
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        config.update(read_json(config_path))
    return config


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-procedural-dataset")


def _build_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_count": len(records),
        "splits": dict(Counter(str(record["split"]) for record in records)),
        "shape_types": dict(Counter(str(record["shape_type"]) for record in records)),
        "colors": dict(Counter(str(record["color"]) for record in records)),
        "sizes": dict(Counter(str(record["size"]) for record in records)),
        "descriptors": dict(Counter(str(record["descriptor"]) for record in records)),
        "filled_voxels": {
            "min": min(int(record["filled_voxels"]) for record in records),
            "max": max(int(record["filled_voxels"]) for record in records),
            "mean": float(np.mean([int(record["filled_voxels"]) for record in records])),
        },
    }


if __name__ == "__main__":
    main()
