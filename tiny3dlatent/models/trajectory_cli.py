from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from tiny3dlatent.models.common import (
    build_vae_from_checkpoint,
    load_checkpoint,
    select_device,
)
from tiny3dlatent.models.flow import ConditionedLatentFlow
from tiny3dlatent.models.metrics import occupancy_from_logits
from tiny3dlatent.representation.cleanup import clean_mesh
from tiny3dlatent.representation.marching_cubes import extract_mesh_from_occupancy
from tiny3dlatent.representation.preview import _draw_mesh
from tiny3dlatent.text.parser import (
    ATTRIBUTE_SIZES,
    attribute_indices,
    parse_prompt,
)
from tiny3dlatent.utils.io import ensure_dir, write_json

DEFAULT_PROMPTS = ["red sphere", "yellow cube", "cyan torus"]


def main() -> None:
    args = _parse_args()
    visualize_trajectory(
        checkpoint_path=args.checkpoint or _latest_flow_checkpoint(),
        prompts=args.prompt or DEFAULT_PROMPTS,
        steps=args.steps,
        frames=args.frames,
        guidance_scale=args.guidance,
        seed=args.seed,
    )


def visualize_trajectory(
    *,
    checkpoint_path: Path,
    prompts: list[str],
    steps: int,
    frames: int,
    guidance_scale: float,
    seed: int,
) -> dict[str, Any]:
    parsed = [parse_prompt(prompt) for prompt in prompts]

    flow_checkpoint = load_checkpoint(checkpoint_path)
    config = flow_checkpoint["config"]
    device = select_device(str(config["device"]))

    vae_checkpoint = load_checkpoint(Path(str(flow_checkpoint["vae_checkpoint"])))
    vae = build_vae_from_checkpoint(vae_checkpoint).to(device)
    vae_config = vae_checkpoint["config"]

    flow = ConditionedLatentFlow(
        latent_dim=int(vae_config["latent_dim"]),
        attribute_sizes=ATTRIBUTE_SIZES,
        attribute_dim=int(config["attribute_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        time_dim=int(config["time_dim"]),
        hidden_layers=int(config["hidden_layers"]),
    ).to(device)
    flow.load_state_dict(flow_checkpoint["model_state"])
    flow.eval()

    latent_mean = flow_checkpoint["latent_mean"].to(device)
    latent_std = flow_checkpoint["latent_std"].to(device)

    # Snapshot the latent at evenly spaced fractions of the integration.
    snapshot_steps = sorted(
        {round(fraction * steps) for fraction in np.linspace(0.0, 1.0, frames)}
    )

    run_dir = _make_run_dir(Path(str(config["output_dir"])))
    rows = []
    stats: dict[str, list[dict[str, Any]]] = {}
    for prompt_index, (prompt, attributes) in enumerate(
        zip(prompts, parsed, strict=True)
    ):
        indices = torch.tensor([attribute_indices(attributes)], device=device)
        nulls = flow.null_attributes(1, device)
        torch.manual_seed(seed + prompt_index)
        z = torch.randn(1, flow.latent_dim, device=device)

        snapshots = []
        with torch.no_grad():
            dt = 1.0 / steps
            for step in range(steps + 1):
                if step in snapshot_steps:
                    snapshots.append((step / steps, z.clone()))
                if step == steps:
                    break
                t = torch.full((1,), step * dt, device=device)
                velocity = flow(z, t, indices)
                if guidance_scale != 1.0:
                    unconditional = flow(z, t, nulls)
                    velocity = unconditional + guidance_scale * (
                        velocity - unconditional
                    )
                z = z + velocity * dt

        frames_out = []
        stats[prompt] = []
        with torch.no_grad():
            for time_fraction, latent in snapshots:
                decoded = vae.decode(latent * latent_std + latent_mean)
                occupancy = (
                    occupancy_from_logits(decoded)[0, 0].cpu().numpy().astype(np.uint8)
                )
                mesh = None
                if occupancy.sum() > 0:
                    mesh = clean_mesh(extract_mesh_from_occupancy(occupancy))
                frames_out.append((time_fraction, mesh))
                stats[prompt].append(
                    {"t": time_fraction, "filled_voxels": int(occupancy.sum())}
                )
        rows.append((prompt, attributes.color or "blue", frames_out))

    image_path = run_dir / "flow_trajectory.png"
    _render_trajectory(rows, image_path)
    write_json(
        run_dir / "metadata.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "flow_checkpoint": checkpoint_path.as_posix(),
            "prompts": prompts,
            "steps": steps,
            "frames": frames,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "voxel_counts": stats,
            "run_dir": run_dir.as_posix(),
        },
    )
    write_json(run_dir / "config.json", dict(config))
    print(f"trajectory strip -> {image_path}")
    for prompt, counts in stats.items():
        progression = " -> ".join(str(item["filled_voxels"]) for item in counts)
        print(f"  {prompt!r} voxels: {progression}")
    return {"run_dir": run_dir.as_posix(), "image": image_path.as_posix()}


def _render_trajectory(
    rows: list[tuple[str, str, list[tuple[float, Any]]]], output_path: Path
) -> None:
    ensure_dir(output_path.parent)
    row_count = len(rows)
    column_count = len(rows[0][2])
    figure = plt.figure(figsize=(column_count * 1.9, row_count * 2.2))
    for row_index, (prompt, color, frames) in enumerate(rows):
        for column, (time_fraction, mesh) in enumerate(frames):
            axis = figure.add_subplot(
                row_count,
                column_count,
                row_index * column_count + column + 1,
                projection="3d",
            )
            if mesh is not None:
                _draw_mesh(axis, mesh, color=color)
            else:
                axis.axis("off")
                axis.text2D(0.5, 0.5, "empty", ha="center", va="center", fontsize=8)
            title = f"t={time_fraction:.2f}"
            if column == 0:
                title = f"{prompt}\n{title}"
            axis.set_title(title, fontsize=7)
    figure.suptitle(
        "rectified flow: noise (t=0) integrates to a shape (t=1)", fontsize=11
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _latest_flow_checkpoint() -> Path:
    for pattern in ("*-color-text-flow/text_flow.pt", "*-text-flow/text_flow.pt"):
        candidates = sorted(Path("outputs/runs").glob(pattern))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(
        "no text-flow checkpoint found under outputs/runs/; train one first"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the flow turning noise into a shape step by step."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a text_flow.pt checkpoint (default: latest color flow).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to visualize (repeatable); defaults to three examples.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Euler steps.")
    parser.add_argument(
        "--frames", type=int, default=8, help="Snapshots along the trajectory."
    )
    parser.add_argument(
        "--guidance", type=float, default=2.0, help="Classifier-free guidance scale."
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-flow-trajectory")


if __name__ == "__main__":
    main()
