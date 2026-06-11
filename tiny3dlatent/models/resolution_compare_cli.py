from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from tiny3dlatent.models.common import (
    build_vae_from_checkpoint,
    load_checkpoint,
    select_device,
)
from tiny3dlatent.models.dataset import OccupancyDataset
from tiny3dlatent.models.metrics import occupancy_from_logits, voxel_iou
from tiny3dlatent.representation.cleanup import clean_mesh
from tiny3dlatent.representation.marching_cubes import extract_mesh_from_occupancy
from tiny3dlatent.representation.preview import _draw_mesh
from tiny3dlatent.utils.io import ensure_dir, write_json


def main() -> None:
    args = _parse_args()
    compare_resolutions(example_count=args.examples)


def compare_resolutions(*, example_count: int) -> dict[str, Any]:
    checkpoint_32 = _latest_vae_for_resolution(32)
    checkpoint_64 = _latest_vae_for_resolution(64)
    device = select_device("auto")

    sides = []
    for checkpoint_path, dataset_dir in (
        (checkpoint_32, "data/procedural"),
        (checkpoint_64, "data/procedural64"),
    ):
        checkpoint = load_checkpoint(checkpoint_path)
        vae = build_vae_from_checkpoint(checkpoint).to(device)
        val_set = OccupancyDataset(Path(dataset_dir), split="val")
        sides.append(
            {
                "checkpoint": checkpoint_path,
                "resolution": int(checkpoint["config"]["resolution"]),
                "vae": vae,
                "val_set": val_set,
            }
        )

    run_dir = _make_run_dir(Path("outputs/runs"))
    columns = []
    ious: dict[str, list[float]] = {"32": [], "64": []}
    with torch.no_grad():
        for example_index in range(example_count):
            row = {"label": str(sides[0]["val_set"].records[example_index]["label"])}
            for side, key in zip(sides, ("32", "64"), strict=True):
                grid, _ = side["val_set"][example_index]
                mean, _ = side["vae"].encode(grid.unsqueeze(0).to(device))
                recon = occupancy_from_logits(side["vae"].decode(mean))
                iou = voxel_iou(recon, grid.unsqueeze(0).to(device).bool())
                ious[key].append(iou)
                occupancy = recon[0, 0].cpu().numpy().astype("uint8")
                row[key] = {
                    "mesh": clean_mesh(extract_mesh_from_occupancy(occupancy)),
                    "iou": iou,
                    "color": str(side["val_set"].records[example_index]["color"]),
                }
            columns.append(row)

    image_path = run_dir / "resolution_comparison.png"
    _render_comparison(columns, image_path)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_32": checkpoint_32.as_posix(),
        "checkpoint_64": checkpoint_64.as_posix(),
        "examples": example_count,
        "mean_recon_iou_32": sum(ious["32"]) / len(ious["32"]),
        "mean_recon_iou_64": sum(ious["64"]) / len(ious["64"]),
        "run_dir": run_dir.as_posix(),
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "config.json", {"examples": example_count})
    print(
        f"recon IoU 32^3 {metadata['mean_recon_iou_32']:.3f} vs "
        f"64^3 {metadata['mean_recon_iou_64']:.3f} -> {image_path}"
    )
    return metadata


def _render_comparison(columns: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    count = len(columns)
    figure = plt.figure(figsize=(count * 2.4, 5.2))
    for index, row in enumerate(columns):
        for row_index, key in enumerate(("32", "64")):
            axis = figure.add_subplot(
                2, count, row_index * count + index + 1, projection="3d"
            )
            entry = row[key]
            _draw_mesh(axis, entry["mesh"], color=entry["color"])
            title = f"{key}^3 (IoU {entry['iou']:.2f})"
            if row_index == 0:
                title = f"{row['label']}\n{title}"
            axis.set_title(title, fontsize=7)
    figure.suptitle("VAE reconstructions: 32^3 (top) vs 64^3 (bottom)", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _latest_vae_for_resolution(resolution: int) -> Path:
    candidates = sorted(Path("outputs/runs").glob("*-vae/vae.pt"))
    matches = []
    for candidate in candidates:
        checkpoint = load_checkpoint(candidate)
        if int(checkpoint["config"]["resolution"]) == resolution:
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f"no VAE checkpoint at resolution {resolution} under outputs/runs/*-vae/"
        )
    return matches[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render side-by-side 32^3 vs 64^3 VAE reconstructions."
    )
    parser.add_argument("--examples", type=int, default=6, help="Examples to compare.")
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-resolution-comparison")


if __name__ == "__main__":
    main()
