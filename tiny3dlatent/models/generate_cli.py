from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.measure import label as connected_components_label

from tiny3dlatent.data.labels import SHAPE_TYPES
from tiny3dlatent.models.common import load_checkpoint, select_device
from tiny3dlatent.models.flow import LatentFlow
from tiny3dlatent.models.metrics import occupancy_from_logits
from tiny3dlatent.models.vae import VAE
from tiny3dlatent.representation.cleanup import clean_mesh
from tiny3dlatent.representation.export import export_mesh
from tiny3dlatent.representation.marching_cubes import extract_mesh_from_occupancy
from tiny3dlatent.representation.preview import save_mesh_preview_grid
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json

CLASS_PREVIEW_COLORS = {
    "sphere": "red",
    "cube": "yellow",
    "rounded_box": "orange",
    "cylinder": "cyan",
    "capsule": "green",
    "torus": "blue",
}
VOXEL_RANGE_MARGIN = (0.5, 1.5)
MESHES_PER_CLASS = 4


def main() -> None:
    args = _parse_args()
    generate(
        checkpoint_path=args.checkpoint or _latest_flow_checkpoint(),
        per_class=args.per_class,
        steps=args.steps,
        seed=args.seed,
    )


def generate(
    *, checkpoint_path: Path, per_class: int, steps: int, seed: int
) -> dict[str, Any]:
    flow_checkpoint = load_checkpoint(checkpoint_path)
    config = flow_checkpoint["config"]
    device = select_device(str(config["device"]))

    vae_checkpoint = load_checkpoint(Path(str(flow_checkpoint["vae_checkpoint"])))
    vae_config = vae_checkpoint["config"]
    vae = VAE(
        resolution=int(vae_config["resolution"]),
        latent_dim=int(vae_config["latent_dim"]),
        base_channels=int(vae_config["base_channels"]),
    ).to(device)
    vae.load_state_dict(vae_checkpoint["model_state"])
    vae.eval()

    model = LatentFlow(
        latent_dim=int(vae_config["latent_dim"]),
        num_classes=len(SHAPE_TYPES),
        hidden_dim=int(config["hidden_dim"]),
        time_dim=int(config["time_dim"]),
        class_dim=int(config["class_dim"]),
        hidden_layers=int(config["hidden_layers"]),
    ).to(device)
    model.load_state_dict(flow_checkpoint["model_state"])
    model.eval()

    latent_mean = flow_checkpoint["latent_mean"].to(device)
    latent_std = flow_checkpoint["latent_std"].to(device)
    voxel_ranges = _class_voxel_ranges(Path(str(config["dataset_dir"])))

    run_dir = _make_run_dir(Path(str(config["output_dir"])))
    mesh_dir = ensure_dir(run_dir / "meshes")

    class_stats: dict[str, dict[str, Any]] = {}
    preview_entries = []
    for class_index, shape_type in enumerate(SHAPE_TYPES):
        class_indices = torch.full((per_class,), class_index, device=device)
        torch.manual_seed(seed + class_index)
        z = model.sample(class_indices, steps=steps)
        z = z * latent_std + latent_mean
        with torch.no_grad():
            logits = vae.decode(z)
        grids = occupancy_from_logits(logits)[:, 0].cpu().numpy().astype(np.uint8)

        low, high = voxel_ranges[shape_type]
        per_sample = []
        mesh_count = 0
        for sample_index, grid in enumerate(grids):
            filled = int(grid.sum())
            sample = {
                "index": sample_index,
                "filled_voxels": filled,
                "components": 0,
                "in_range": False,
                "mesh_file": None,
            }
            if filled > 0:
                sample["components"] = int(
                    connected_components_label(grid, connectivity=1).max()
                )
                sample["in_range"] = bool(low <= filled <= high)
                if mesh_count < MESHES_PER_CLASS:
                    mesh = clean_mesh(extract_mesh_from_occupancy(grid))
                    obj_path = mesh_dir / f"{shape_type}_{sample_index:02d}.obj"
                    export_mesh(mesh, obj_path)
                    sample["mesh_file"] = obj_path.as_posix()
                    preview_entries.append(
                        (
                            mesh,
                            {
                                "label": f"generated {shape_type} #{sample_index}",
                                "color": CLASS_PREVIEW_COLORS[shape_type],
                            },
                        )
                    )
                    mesh_count += 1
            per_sample.append(sample)

        non_empty = sum(1 for s in per_sample if s["filled_voxels"] > 0)
        single_component = sum(1 for s in per_sample if s["components"] == 1)
        in_range = sum(1 for s in per_sample if s["in_range"])
        class_stats[shape_type] = {
            "samples": per_class,
            "non_empty": non_empty,
            "single_component": single_component,
            "voxel_count_in_range": in_range,
            "expected_voxel_range": [low, high],
            "voxel_counts": [s["filled_voxels"] for s in per_sample],
            "per_sample": per_sample,
        }
        print(
            f"{shape_type:12s} non_empty {non_empty}/{per_class}  "
            f"single_component {single_component}/{per_class}  "
            f"in_range {in_range}/{per_class}"
        )

    save_mesh_preview_grid(
        preview_entries, run_dir / "generated_mesh_grid.png", columns=MESHES_PER_CLASS
    )

    totals = {
        "samples": per_class * len(SHAPE_TYPES),
        "non_empty": sum(s["non_empty"] for s in class_stats.values()),
        "single_component": sum(s["single_component"] for s in class_stats.values()),
        "voxel_count_in_range": sum(
            s["voxel_count_in_range"] for s in class_stats.values()
        ),
    }
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "flow_checkpoint": checkpoint_path.as_posix(),
        "vae_checkpoint": str(flow_checkpoint["vae_checkpoint"]),
        "per_class": per_class,
        "steps": steps,
        "seed": seed,
        "run_dir": run_dir.as_posix(),
        "totals": totals,
    }
    write_json(run_dir / "generation_stats.json", class_stats)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "config.json", dict(config))

    print(
        f"total non_empty {totals['non_empty']}/{totals['samples']}  "
        f"single_component {totals['single_component']}/{totals['samples']}  "
        f"in_range {totals['voxel_count_in_range']}/{totals['samples']} -> {run_dir}"
    )
    return {"run_dir": run_dir.as_posix(), "class_stats": class_stats, "totals": totals}


def _class_voxel_ranges(dataset_dir: Path) -> dict[str, tuple[float, float]]:
    metadata = read_json(dataset_dir / "metadata.json")
    by_type: dict[str, list[int]] = {name: [] for name in SHAPE_TYPES}
    for record in metadata["records"]:
        if record["split"] == "train":
            by_type[str(record["shape_type"])].append(int(record["filled_voxels"]))
    low_margin, high_margin = VOXEL_RANGE_MARGIN
    return {
        name: (low_margin * min(counts), high_margin * max(counts))
        for name, counts in by_type.items()
    }


def _latest_flow_checkpoint() -> Path:
    candidates = sorted(Path("outputs/runs").glob("*-latent-flow/flow.pt"))
    if not candidates:
        raise FileNotFoundError(
            "no flow checkpoint found under outputs/runs/*-latent-flow/; train one "
            "with python -m tiny3dlatent.models.train_flow"
        )
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate shapes from the latent flow and decode them to meshes."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a flow.pt checkpoint (default: latest under outputs/runs).",
    )
    parser.add_argument(
        "--per-class", type=int, default=16, help="Samples per shape class."
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Euler integration steps."
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-latent-generation")


if __name__ == "__main__":
    main()
