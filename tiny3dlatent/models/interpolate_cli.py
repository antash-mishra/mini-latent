from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from tiny3dlatent.models.common import load_checkpoint, select_device
from tiny3dlatent.models.dataset import OccupancyDataset
from tiny3dlatent.models.metrics import occupancy_from_logits
from tiny3dlatent.models.vae import VAE
from tiny3dlatent.utils.io import ensure_dir, write_json

DEFAULT_PAIRS = [
    ("cube", "cylinder"),
    ("sphere", "cube"),
    ("torus", "capsule"),
]


def main() -> None:
    args = _parse_args()
    interpolate(
        checkpoint_path=args.checkpoint or _latest_vae_checkpoint(),
        steps=args.steps,
    )


def interpolate(*, checkpoint_path: Path, steps: int) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    device = select_device(str(config["device"]))
    model = VAE(
        resolution=int(config["resolution"]),
        latent_dim=int(config["latent_dim"]),
        base_channels=int(config["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_set = OccupancyDataset(Path(str(config["dataset_dir"])), split="val")
    by_type: dict[str, int] = {}
    for index, record in enumerate(val_set.records):
        by_type.setdefault(str(record["shape_type"]), index)

    run_dir = _make_run_dir(Path(str(config["output_dir"])))
    rows = []
    row_labels = []
    with torch.no_grad():
        for start_type, end_type in DEFAULT_PAIRS:
            if start_type not in by_type or end_type not in by_type:
                continue
            start_grid, _ = val_set[by_type[start_type]]
            end_grid, _ = val_set[by_type[end_type]]
            start_mean, _ = model.encode(start_grid.unsqueeze(0).to(device))
            end_mean, _ = model.encode(end_grid.unsqueeze(0).to(device))
            frames = []
            for step in range(steps):
                alpha = step / (steps - 1)
                z = (1.0 - alpha) * start_mean + alpha * end_mean
                occupancy = occupancy_from_logits(model.decode(z))[0, 0].cpu().numpy()
                frames.append(occupancy)
            rows.append(frames)
            row_labels.append(f"{start_type} -> {end_type}")

    strip_path = run_dir / "interpolation_strip.png"
    _save_strip(rows, row_labels, strip_path)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": checkpoint_path.as_posix(),
        "steps": steps,
        "pairs": row_labels,
        "run_dir": run_dir.as_posix(),
        "nonempty_frames": [[bool(frame.any()) for frame in frames] for frames in rows],
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "config.json", dict(config))
    print(f"saved {len(rows)} interpolation rows -> {strip_path}")
    return metadata


def _save_strip(
    rows: list[list[np.ndarray]], row_labels: list[str], output_path: Path
) -> None:
    if not rows:
        raise ValueError("no interpolation rows to render")
    ensure_dir(output_path.parent)
    steps = len(rows[0])
    figure, axes = plt.subplots(
        len(rows), steps, figsize=(steps * 1.6, len(rows) * 1.9)
    )
    axes = np.atleast_2d(axes)
    for row_index, frames in enumerate(rows):
        for column, frame in enumerate(frames):
            axis = axes[row_index, column]
            axis.imshow(
                frame[:, frame.shape[1] // 2, :],
                cmap="gray_r",
                interpolation="nearest",
            )
            axis.axis("off")
            if column == 0:
                axis.set_title(row_labels[row_index], fontsize=8, loc="left")
    figure.suptitle("latent interpolation (middle slices)", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _latest_vae_checkpoint() -> Path:
    candidates = sorted(Path("outputs/runs").glob("*-vae/vae.pt"))
    if not candidates:
        raise FileNotFoundError(
            "no VAE checkpoint found under outputs/runs/*-vae/; train one with "
            "python -m tiny3dlatent.models.train_vae"
        )
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate between two shapes in VAE latent space."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a vae.pt checkpoint (default: latest under outputs/runs).",
    )
    parser.add_argument(
        "--steps", type=int, default=8, help="Number of interpolation frames."
    )
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-vae-interpolation")


if __name__ == "__main__":
    main()
