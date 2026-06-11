from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import trimesh
from torch.utils.data import DataLoader

from tiny3dlatent.data.labels import SHAPE_TYPES
from tiny3dlatent.models.common import (
    build_vae_from_checkpoint,
    load_checkpoint,
    select_device,
)
from tiny3dlatent.models.dataset import OccupancyDataset
from tiny3dlatent.models.generate_cli import (
    CLASS_PREVIEW_COLORS,
    generate,
)
from tiny3dlatent.models.metrics import occupancy_from_logits
from tiny3dlatent.models.recon_preview import save_recon_grid
from tiny3dlatent.models.text_generate_cli import generate_from_prompts
from tiny3dlatent.report.html import render_html_report
from tiny3dlatent.representation.preview import save_turntable_strip
from tiny3dlatent.representation.stats import mesh_stats
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json

DEFAULT_CONFIG = {
    "output_dir": "outputs/runs",
    "dataset_dir": "data/procedural",
    "device": "auto",
    "per_class": 16,
    "steps": 50,
    "seed": 0,
    "prompts": [
        "red sphere",
        "yellow large cube",
        "blue tall cylinder",
        "green wide rounded box",
    ],
    "per_prompt": 4,
    "guidance_scale": 2.0,
    "extreme_recon_count": 4,
    "turntable_frames": 8,
}


def main() -> None:
    args = _parse_args()
    config = DEFAULT_CONFIG.copy()
    if args.config and args.config.exists():
        config.update(read_json(args.config))
    build_report(config)


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(str(config["output_dir"]))
    run_dir = _make_run_dir(output_root)
    image_dir = ensure_dir(run_dir / "images")

    recon = _reconstruction_section(config, image_dir)
    generation = _class_generation_section(config, image_dir, run_dir)
    prompts = _prompt_section(config, image_dir)
    failures = _failure_section(recon, generation)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir.as_posix(),
        "config": config,
        "reconstruction": recon["stats"],
        "class_generation": generation["stats"],
        "mesh_quality": generation["mesh_quality"],
        "prompt_generation": prompts["stats"],
        "failures": failures,
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", config)

    html_path = run_dir / "generation_report.html"
    html_path.write_text(
        render_html_report(
            report,
            recon_images=recon["images"],
            generation_images=generation["images"],
            prompt_images=prompts["images"],
        )
    )
    print(f"report -> {html_path}")
    return {"run_dir": run_dir.as_posix(), "html": html_path.as_posix()}


def _reconstruction_section(config: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    """Per-example val IoU for the latest VAE, plus best/worst recon images."""
    checkpoint_path = sorted(Path(str(config["output_dir"])).glob("*-vae/vae.pt"))[-1]
    checkpoint = load_checkpoint(checkpoint_path)
    device = select_device(str(config["device"]))
    vae = build_vae_from_checkpoint(checkpoint).to(device)

    val_set = OccupancyDataset(Path(str(config["dataset_dir"])), split="val")
    loader = DataLoader(val_set, batch_size=32)
    per_example: list[dict[str, Any]] = []
    with torch.no_grad():
        offset = 0
        for grids, _ in loader:
            grids = grids.to(device)
            mean, _ = vae.encode(grids)
            recon = occupancy_from_logits(vae.decode(mean))
            pred = recon.flatten(start_dim=1)
            targ = grids.bool().flatten(start_dim=1)
            intersection = (pred & targ).sum(dim=1).float()
            union = (pred | targ).sum(dim=1).float().clamp_min(1.0)
            for batch_index, iou in enumerate((intersection / union).tolist()):
                record = val_set.records[offset + batch_index]
                per_example.append(
                    {
                        "index": offset + batch_index,
                        "iou": iou,
                        "label": record["label"],
                    }
                )
            offset += grids.shape[0]

    ranked = sorted(per_example, key=lambda item: item["iou"])
    count = int(config["extreme_recon_count"])
    worst, best = ranked[:count], ranked[-count:]

    images = {}
    for name, subset in (("worst_recons", worst), ("best_recons", best)):
        entries = []
        with torch.no_grad():
            for item in subset:
                grid, _ = val_set[item["index"]]
                mean, _ = vae.encode(grid.unsqueeze(0).to(device))
                recon_grid = occupancy_from_logits(vae.decode(mean))[0, 0].cpu().numpy()
                record = dict(val_set.records[item["index"]])
                record["label"] = f"{record['label']} (IoU {item['iou']:.2f})"
                entries.append((grid[0].numpy(), recon_grid, record))
        path = image_dir / f"{name}.png"
        save_recon_grid(entries, path)
        images[name] = path.name

    ious = [item["iou"] for item in per_example]
    return {
        "stats": {
            "checkpoint": checkpoint_path.as_posix(),
            "val_examples": len(per_example),
            "mean_iou": sum(ious) / len(ious),
            "min_iou": min(ious),
            "max_iou": max(ious),
            "worst": worst,
            "best": best,
        },
        "images": images,
    }


def _class_generation_section(
    config: dict[str, Any], image_dir: Path, run_dir: Path
) -> dict[str, Any]:
    """Fresh class-conditional samples with checks, renders, and turntables."""
    checkpoint_path = sorted(
        Path(str(config["output_dir"])).glob("*-latent-flow/flow.pt")
    )[-1]
    result = generate(
        checkpoint_path=checkpoint_path,
        per_class=int(config["per_class"]),
        steps=int(config["steps"]),
        seed=int(config["seed"]),
    )
    generation_run = Path(result["run_dir"])

    images = {}
    grid_source = generation_run / "generated_mesh_grid.png"
    if grid_source.exists():
        shutil.copy(grid_source, image_dir / "generated_mesh_grid.png")
        images["mesh_grid"] = "generated_mesh_grid.png"

    turntables = []
    mesh_quality = []
    for shape_type in SHAPE_TYPES:
        per_sample = result["class_stats"][shape_type]["per_sample"]
        first_mesh = next((s["mesh_file"] for s in per_sample if s["mesh_file"]), None)
        if first_mesh is None:
            continue
        mesh = trimesh.load(first_mesh, process=False)
        strip_name = f"turntable_{shape_type}.png"
        save_turntable_strip(
            mesh,
            image_dir / strip_name,
            title=f"generated {shape_type}",
            color=CLASS_PREVIEW_COLORS[shape_type],
            frames=int(config["turntable_frames"]),
        )
        turntables.append(strip_name)

        stats = mesh_stats(mesh, Path(first_mesh))
        bounds_ok = all(
            -1.1 <= value <= 1.1
            for value in [*stats["bounds_min"], *stats["bounds_max"]]
        )
        mesh_quality.append(
            {
                "shape_type": shape_type,
                "vertices": stats["vertex_count"],
                "faces": stats["face_count"],
                "watertight": stats["is_watertight"],
                "components": stats["connected_components"],
                "bounds_ok": bounds_ok,
            }
        )
    images["turntables"] = turntables

    return {
        "stats": {
            "checkpoint": checkpoint_path.as_posix(),
            "generation_run": generation_run.as_posix(),
            "totals": result["totals"],
            "per_class": {
                name: {
                    key: value
                    for key, value in stats.items()
                    if key not in ("per_sample", "voxel_counts")
                }
                for name, stats in result["class_stats"].items()
            },
            "per_sample": {
                name: stats["per_sample"]
                for name, stats in result["class_stats"].items()
            },
        },
        "mesh_quality": mesh_quality,
        "images": images,
    }


def _prompt_section(config: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    checkpoint_path = sorted(
        Path(str(config["output_dir"])).glob("*-text-flow/text_flow.pt")
    )[-1]
    result = generate_from_prompts(
        checkpoint_path=checkpoint_path,
        prompts=list(config["prompts"]),
        per_prompt=int(config["per_prompt"]),
        steps=int(config["steps"]),
        guidance_scale=float(config["guidance_scale"]),
        seed=int(config["seed"]),
    )
    gallery = Path(result["run_dir"]) / "prompt_gallery.png"
    images = {}
    if gallery.exists():
        shutil.copy(gallery, image_dir / "prompt_gallery.png")
        images["gallery"] = "prompt_gallery.png"
    return {
        "stats": {
            "checkpoint": checkpoint_path.as_posix(),
            "prompt_run": result["run_dir"],
            "prompts": result["prompt_stats"],
        },
        "images": images,
    }


def _failure_section(
    recon: dict[str, Any], generation: dict[str, Any]
) -> dict[str, Any]:
    """Collect everything that failed a check, with notes for the gallery."""
    failures: list[dict[str, str]] = []
    for item in recon["stats"]["worst"]:
        failures.append(
            {
                "kind": "reconstruction",
                "subject": item["label"],
                "note": (
                    f"lowest val IoU {item['iou']:.2f}: thin/small features are "
                    "the first thing the 32^3 bottleneck loses"
                ),
            }
        )
    for shape_type, samples in generation["stats"]["per_sample"].items():
        for sample in samples:
            notes = []
            if sample["filled_voxels"] == 0:
                notes.append("empty grid")
            elif sample["components"] != 1:
                notes.append(f"{sample['components']} connected components")
            if sample["filled_voxels"] > 0 and not sample["in_range"]:
                notes.append(
                    f"voxel count {sample['filled_voxels']} outside class range"
                )
            if notes:
                failures.append(
                    {
                        "kind": "generation",
                        "subject": f"{shape_type} sample #{sample['index']}",
                        "note": "; ".join(notes),
                    }
                )
    generation_failures = [f for f in failures if f["kind"] == "generation"]
    return {
        "items": failures,
        "generation_failure_count": len(generation_failures),
        "note": (
            "no generated sample failed the structural checks this run"
            if not generation_failures
            else f"{len(generation_failures)} generated samples failed checks"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the HTML generation report end-to-end."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/report.json"),
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-generation-report")


if __name__ == "__main__":
    main()
