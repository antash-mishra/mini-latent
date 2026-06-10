from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny3dlatent.models.common import load_checkpoint, select_device
from tiny3dlatent.models.flow import ConditionedLatentFlow
from tiny3dlatent.models.metrics import occupancy_from_logits
from tiny3dlatent.models.vae import VAE
from tiny3dlatent.representation.cleanup import clean_mesh
from tiny3dlatent.representation.export import export_mesh
from tiny3dlatent.representation.marching_cubes import extract_mesh_from_occupancy
from tiny3dlatent.representation.preview import save_mesh_preview_grid
from tiny3dlatent.text.parser import (
    ATTRIBUTE_SIZES,
    attribute_indices,
    parse_prompt,
)
from tiny3dlatent.utils.io import ensure_dir, write_json

DEFAULT_PROMPTS = [
    "red sphere",
    "yellow large cube",
    "blue tall cylinder",
    "green wide rounded box",
    "orange small capsule",
    "cyan torus",
]
ASPECT_TOLERANCE = 1.05


def main() -> None:
    args = _parse_args()
    generate_from_prompts(
        checkpoint_path=args.checkpoint or _latest_text_flow_checkpoint(),
        prompts=args.prompt or DEFAULT_PROMPTS,
        per_prompt=args.per_prompt,
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )


def generate_from_prompts(
    *,
    checkpoint_path: Path,
    prompts: list[str],
    per_prompt: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> dict[str, Any]:
    parsed = [parse_prompt(prompt) for prompt in prompts]

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

    model = ConditionedLatentFlow(
        latent_dim=int(vae_config["latent_dim"]),
        attribute_sizes=ATTRIBUTE_SIZES,
        attribute_dim=int(config["attribute_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        time_dim=int(config["time_dim"]),
        hidden_layers=int(config["hidden_layers"]),
    ).to(device)
    model.load_state_dict(flow_checkpoint["model_state"])
    model.eval()

    latent_mean = flow_checkpoint["latent_mean"].to(device)
    latent_std = flow_checkpoint["latent_std"].to(device)

    run_dir = _make_run_dir(Path(str(config["output_dir"])))
    mesh_dir = ensure_dir(run_dir / "meshes")

    prompt_stats: dict[str, dict[str, Any]] = {}
    preview_entries = []
    for prompt_index, (prompt, attributes) in enumerate(
        zip(prompts, parsed, strict=True)
    ):
        indices = torch.tensor(
            [attribute_indices(attributes)] * per_prompt, device=device
        )
        torch.manual_seed(seed + prompt_index)
        z = model.sample(indices, steps=steps, guidance_scale=guidance_scale)
        z = z * latent_std + latent_mean
        with torch.no_grad():
            logits = vae.decode(z)
        grids = occupancy_from_logits(logits)[:, 0].cpu().numpy().astype(np.uint8)

        slug = prompt.replace(" ", "_")
        aspects = []
        non_empty = 0
        for sample_index, grid in enumerate(grids):
            if grid.sum() == 0:
                continue
            non_empty += 1
            aspects.append(_aspect_ratio(grid))
            mesh = clean_mesh(extract_mesh_from_occupancy(grid))
            mesh_path = mesh_dir / f"{slug}_{sample_index:02d}.obj"
            export_mesh(mesh, mesh_path)
            write_json(
                mesh_path.with_suffix(".prompt.json"),
                {
                    "prompt": prompt,
                    "parsed_attributes": attributes.to_metadata(),
                    "guidance_scale": guidance_scale,
                    "steps": steps,
                    "seed": seed + prompt_index,
                    "sample_index": sample_index,
                    "mesh_file": mesh_path.as_posix(),
                },
            )
            if sample_index < 2:
                preview_entries.append(
                    (mesh, {"label": prompt, "color": attributes.color or "blue"})
                )

        aspect_check = _check_aspect(attributes.descriptor, aspects)
        prompt_stats[prompt] = {
            "samples": per_prompt,
            "non_empty": non_empty,
            "parsed_attributes": attributes.to_metadata(),
            "aspect_ratios": aspects,
            "aspect_check": aspect_check,
        }
        check_note = "" if aspect_check is None else f"  aspect_ok {aspect_check}"
        print(f"{prompt!r}: non_empty {non_empty}/{per_prompt}{check_note}")

    save_mesh_preview_grid(preview_entries, run_dir / "prompt_gallery.png", columns=4)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "flow_checkpoint": checkpoint_path.as_posix(),
        "vae_checkpoint": str(flow_checkpoint["vae_checkpoint"]),
        "prompts": prompts,
        "per_prompt": per_prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "run_dir": run_dir.as_posix(),
    }
    write_json(run_dir / "prompt_stats.json", prompt_stats)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "config.json", dict(config))
    print(f"prompt gallery -> {run_dir / 'prompt_gallery.png'}")
    return {"run_dir": run_dir.as_posix(), "prompt_stats": prompt_stats}


def _aspect_ratio(grid: np.ndarray) -> float:
    """Height/width ratio of the occupied bounding box (y is the height axis)."""
    occupied = np.argwhere(grid > 0)
    extents = occupied.max(axis=0) - occupied.min(axis=0) + 1
    horizontal = (extents[0] + extents[2]) / 2.0
    return float(extents[1] / horizontal)


def _check_aspect(descriptor: str | None, aspects: list[float]) -> str | None:
    """For tall/wide prompts: are most samples stretched the right way?"""
    if descriptor not in ("tall", "wide") or not aspects:
        return None
    if descriptor == "tall":
        passed = sum(1 for aspect in aspects if aspect > ASPECT_TOLERANCE)
    else:
        passed = sum(1 for aspect in aspects if aspect < 1.0 / ASPECT_TOLERANCE)
    return f"{passed}/{len(aspects)}"


def _latest_text_flow_checkpoint() -> Path:
    candidates = sorted(Path("outputs/runs").glob("*-text-flow/text_flow.pt"))
    if not candidates:
        raise FileNotFoundError(
            "no text-flow checkpoint found under outputs/runs/*-text-flow/; train "
            "one with python -m tiny3dlatent.models.train_text_flow"
        )
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate meshes from tiny-vocabulary text prompts."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a text_flow.pt checkpoint (default: latest under outputs/runs).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to generate (repeatable); defaults to a small gallery.",
    )
    parser.add_argument("--per-prompt", type=int, default=4, help="Samples per prompt.")
    parser.add_argument("--steps", type=int, default=50, help="Euler steps.")
    parser.add_argument(
        "--guidance", type=float, default=2.0, help="Classifier-free guidance scale."
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-text-generation")


if __name__ == "__main__":
    main()
