from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny3dlatent.data.labels import COLOR_RGB, DESCRIPTOR_MATERIAL
from tiny3dlatent.models.common import (
    build_vae_from_checkpoint,
    load_checkpoint,
    select_device,
)
from tiny3dlatent.models.flow import ConditionedLatentFlow
from tiny3dlatent.models.metrics import occupancy_from_logits
from tiny3dlatent.representation.cleanup import clean_mesh
from tiny3dlatent.representation.export import export_glb_with_material
from tiny3dlatent.representation.marching_cubes import extract_mesh_from_occupancy
from tiny3dlatent.representation.preview import save_mesh_preview_grid
from tiny3dlatent.text.parser import (
    ATTRIBUTE_SIZES,
    attribute_indices,
    parse_prompt,
)
from tiny3dlatent.utils.io import ensure_dir, write_json

DEFAULT_PROMPTS = [
    "red metallic sphere",
    "green matte cube",
    "blue metallic cylinder",
    "yellow matte rounded box",
    "orange metallic capsule",
    "cyan matte torus",
]


def main() -> None:
    args = _parse_args()
    generate_assets(
        checkpoint_path=args.checkpoint or _latest_color_flow_checkpoint(),
        prompts=args.prompt or DEFAULT_PROMPTS,
        per_prompt=args.per_prompt,
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )


def generate_assets(
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
    vae = build_vae_from_checkpoint(vae_checkpoint).to(device)
    if not hasattr(vae, "decode_full"):
        raise ValueError(
            "asset generation needs a color VAE checkpoint; train one with "
            "python -m tiny3dlatent.models.train_color_vae and retrain the flow "
            "with vae_checkpoint set to latest-color"
        )
    vae_config = vae_checkpoint["config"]

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
    asset_dir = ensure_dir(run_dir / "assets")

    asset_stats: dict[str, dict[str, Any]] = {}
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
            logits, rgb_grids, materials = vae.decode_full(z)
        occupancies = occupancy_from_logits(logits)[:, 0].cpu().numpy().astype(np.uint8)
        rgb_grids = rgb_grids.cpu().numpy()
        materials = materials.cpu().numpy()

        slug = prompt.replace(" ", "_")
        samples = []
        for sample_index, occupancy in enumerate(occupancies):
            if occupancy.sum() == 0:
                continue
            mask = occupancy.astype(bool)
            base_color = tuple(
                float(channel[mask].mean()) for channel in rgb_grids[sample_index]
            )
            roughness = float(materials[sample_index][0])
            metallic = float(materials[sample_index][1])

            mesh = clean_mesh(extract_mesh_from_occupancy(occupancy))
            glb_path = asset_dir / f"{slug}_{sample_index:02d}.glb"
            export_glb_with_material(
                mesh,
                glb_path,
                base_color=base_color,
                roughness=roughness,
                metallic=metallic,
                name=prompt,
            )
            material_summary = {
                "prompt": prompt,
                "parsed_attributes": attributes.to_metadata(),
                "predicted_base_color": list(base_color),
                "predicted_roughness": roughness,
                "predicted_metallic": metallic,
                "guidance_scale": guidance_scale,
                "steps": steps,
                "seed": seed + prompt_index,
                "glb_file": glb_path.as_posix(),
            }
            write_json(glb_path.with_suffix(".material.json"), material_summary)
            samples.append(material_summary)
            if sample_index < 2:
                preview_entries.append(
                    (mesh, {"label": prompt, "rgb": list(base_color)})
                )

        asset_stats[prompt] = {
            "samples": per_prompt,
            "exported": len(samples),
            "assets": samples,
        }
        if samples:
            first = samples[0]
            print(
                f"{prompt!r}: exported {len(samples)}/{per_prompt}  "
                f"color {tuple(round(v, 2) for v in first['predicted_base_color'])}  "
                f"rough {first['predicted_roughness']:.2f}  "
                f"metal {first['predicted_metallic']:.2f}"
            )

    save_mesh_preview_grid(preview_entries, run_dir / "asset_gallery.png", columns=4)
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
        "palette": {name: list(value) for name, value in COLOR_RGB.items()},
        "descriptor_materials": DESCRIPTOR_MATERIAL,
    }
    write_json(run_dir / "asset_stats.json", asset_stats)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "config.json", dict(config))
    print(f"asset gallery -> {run_dir / 'asset_gallery.png'}")
    return {"run_dir": run_dir.as_posix(), "asset_stats": asset_stats}


def _latest_color_flow_checkpoint() -> Path:
    candidates = sorted(Path("outputs/runs").glob("*-color-text-flow/text_flow.pt"))
    if not candidates:
        raise FileNotFoundError(
            "no color text-flow checkpoint found under outputs/runs/*-color-text-flow/; "
            "train one with python -m tiny3dlatent.models.train_text_flow --config "
            "configs/color_text_flow.json"
        )
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate colored GLB assets from text prompts."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a text_flow.pt trained on a color VAE (default: latest).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to generate (repeatable); defaults to a material gallery.",
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
    return ensure_dir(output_root / f"{timestamp}-asset-generation")


if __name__ == "__main__":
    main()
