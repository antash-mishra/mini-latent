from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tiny3dlatent.representation.cleanup import clean_mesh
from tiny3dlatent.representation.export import export_mesh
from tiny3dlatent.representation.marching_cubes import extract_mesh_from_occupancy
from tiny3dlatent.representation.preview import save_mesh_preview_grid, save_mesh_views
from tiny3dlatent.representation.stats import mesh_stats
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json

DEFAULT_CONFIG = {
    "dataset_dir": "data/procedural",
    "output_dir": "outputs/runs",
    "examples_per_shape_type": 2,
    "split": "val",
    "iso_value": 0.5,
    "pad_width": 1,
}


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    extract_meshes(config)


def extract_meshes(config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(str(config["dataset_dir"]))
    output_root = Path(str(config["output_dir"]))
    per_type = int(config["examples_per_shape_type"])
    split = str(config["split"])
    iso_value = float(config["iso_value"])
    pad_width = int(config["pad_width"])

    records = _select_records(dataset_dir, split=split, per_type=per_type)
    run_dir = _make_run_dir(output_root)
    mesh_dir = ensure_dir(run_dir / "meshes")

    all_stats: dict[str, dict[str, object]] = {}
    preview_entries = []
    first_entry = None

    for record in records:
        occupancy = np.load(record["grid_file"])
        mesh = clean_mesh(
            extract_mesh_from_occupancy(
                occupancy, iso_value=iso_value, pad_width=pad_width
            )
        )
        obj_path = export_mesh(mesh, mesh_dir / f"{record['id']}.obj")

        stats = mesh_stats(mesh, obj_path)
        stats["label"] = record["label"]
        stats["shape_type"] = record["shape_type"]
        stats["source_grid"] = record["grid_file"]
        stats["mesh_file"] = obj_path.as_posix()
        all_stats[str(record["id"])] = stats

        preview_entries.append((mesh, record))
        if first_entry is None:
            first_entry = (mesh, record)

    write_json(run_dir / "mesh_stats.json", all_stats)
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "metadata.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": config,
            "dataset_dir": dataset_dir.as_posix(),
            "run_dir": run_dir.as_posix(),
            "mesh_count": len(all_stats),
            "watertight_count": sum(
                1 for stats in all_stats.values() if stats["is_watertight"]
            ),
        },
    )

    save_mesh_preview_grid(preview_entries, run_dir / "mesh_preview_grid.png")
    if first_entry is not None:
        first_mesh, first_record = first_entry
        save_mesh_views(
            first_mesh,
            run_dir / "mesh_views.png",
            title=str(first_record["label"]),
            color=str(first_record["color"]),
        )

    watertight = sum(1 for stats in all_stats.values() if stats["is_watertight"])
    print(f"extracted {len(all_stats)} meshes ({watertight} watertight) -> {run_dir}")
    return {"run_dir": run_dir.as_posix(), "stats": all_stats}


def _select_records(
    dataset_dir: Path, *, split: str, per_type: int
) -> list[dict[str, Any]]:
    metadata = read_json(dataset_dir / "metadata.json")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in metadata["records"]:
        if record["split"] == split and len(by_type[record["shape_type"]]) < per_type:
            by_type[record["shape_type"]].append(record)
    return [record for records in by_type.values() for record in records]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract meshes from procedural occupancy grids."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mesh_extraction.json"),
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
    return ensure_dir(output_root / f"{timestamp}-mesh-extraction")


if __name__ == "__main__":
    main()
