from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tiny3dlatent.data.preview import save_example_slices, save_preview_grid
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json


def main() -> None:
    args = _parse_args()
    preview_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        count=args.count,
    )


def preview_dataset(
    dataset_dir: Path,
    output_dir: Path | None = None,
    count: int = 20,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    metadata = read_json(metadata_path)
    records = metadata.get("records", [])
    if not records:
        raise ValueError("No records found in dataset metadata")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs/runs") / f"{timestamp}-preview"
    output_dir = ensure_dir(output_dir)

    preview_records = records[:count]
    examples: list[tuple[np.ndarray, dict[str, object]]] = []
    first_example: tuple[np.ndarray, dict[str, object]] | None = None

    for record in preview_records:
        grid_file = Path(record["grid_file"])
        if not grid_file.is_absolute():
            # Prefer reconstructing from split/id for portability
            grid_file = dataset_dir / str(record["split"]) / f"{record['id']}_occupancy.npy"
            if not grid_file.exists():
                grid_file = Path(record["grid_file"])
        occupancy = np.load(grid_file)
        examples.append((occupancy, record))
        if first_example is None:
            first_example = (occupancy, record)

    if examples:
        save_preview_grid(examples, output_dir / "preview_grid.png")
    if first_example is not None:
        first_grid, first_metadata = first_example
        save_example_slices(
            first_grid,
            output_dir / "example_slices.png",
            title=str(first_metadata["label"]),
            color=str(first_metadata["color"]),
        )

    preview_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": dataset_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "preview_count": len(examples),
        "first_label": str(first_example[1]["label"]) if first_example else None,
    }
    write_json(output_dir / "preview_metadata.json", preview_metadata)

    print(f"Preview written to {output_dir}")
    return preview_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview an existing procedural dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/procedural"),
        help="Path to the dataset directory containing metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write preview images. Defaults to a timestamped run folder.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of examples to include in the preview grid.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
