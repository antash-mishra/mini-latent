from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from tiny3dlatent.models.autoencoder import Autoencoder
from tiny3dlatent.models.common import (
    count_parameters,
    save_checkpoint,
    select_device,
    set_seed,
)
from tiny3dlatent.models.dataset import OccupancyDataset
from tiny3dlatent.models.metrics import occupancy_from_logits, voxel_iou
from tiny3dlatent.models.recon_preview import save_recon_grid
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json

DEFAULT_CONFIG = {
    "seed": 0,
    "dataset_dir": "data/procedural",
    "output_dir": "outputs/runs",
    "device": "auto",
    "resolution": 32,
    "latent_dim": 128,
    "base_channels": 16,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "pos_weight": 3.0,
    "epochs": 40,
    "overfit_count": 0,
    "preview_count": 8,
}


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.overfit is not None:
        config["overfit_count"] = args.overfit
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train_autoencoder(config)


def train_autoencoder(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))
    overfit_count = int(config["overfit_count"])

    dataset_dir = Path(str(config["dataset_dir"]))
    train_set = OccupancyDataset(
        dataset_dir, split="train", limit=overfit_count or None
    )
    val_set = train_set if overfit_count else OccupancyDataset(dataset_dir, split="val")

    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = Autoencoder(
        resolution=int(config["resolution"]),
        latent_dim=int(config["latent_dim"]),
        base_channels=int(config["base_channels"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    pos_weight = torch.tensor(float(config.get("pos_weight", 1.0)), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    experiment = "autoencoder-overfit" if overfit_count else "autoencoder"
    run_dir = _make_run_dir(Path(str(config["output_dir"])), experiment)
    started = time.time()

    history: list[dict[str, float]] = []
    best_val_iou = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    epochs = int(config["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for grids, _ in train_loader:
            grids = grids.to(device)
            logits, _ = model(grids)
            loss = loss_fn(logits, grids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * grids.shape[0]
        train_loss /= len(train_set)

        val_loss, val_iou = _evaluate(model, val_loader, loss_fn, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }
        )
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % max(1, epochs // 20) == 0 or epoch == epochs:
            print(
                f"epoch {epoch:4d}/{epochs}  train_loss {train_loss:.4f}  "
                f"val_loss {val_loss:.4f}  val_iou {val_iou:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(
        run_dir / "autoencoder.pt",
        model,
        config=config,
        epoch=best_epoch,
        val_iou=best_val_iou,
    )

    preview_count = min(int(config["preview_count"]), len(val_set))
    _save_recon_preview(
        model, val_set, run_dir / "recon_grid.png", preview_count, device
    )

    runtime = time.time() - started
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "run_dir": run_dir.as_posix(),
        "device": str(device),
        "parameter_count": count_parameters(model),
        "train_examples": len(train_set),
        "val_examples": len(val_set),
        "overfit_mode": bool(overfit_count),
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "final_train_loss": history[-1]["train_loss"],
        "runtime_seconds": runtime,
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metadata.json", metadata)

    print(
        f"best val_iou {best_val_iou:.4f} (epoch {best_epoch}) "
        f"in {runtime:.1f}s -> {run_dir}"
    )
    return {
        "run_dir": run_dir.as_posix(),
        "best_val_iou": best_val_iou,
        "final_train_loss": history[-1]["train_loss"],
        "history": history,
    }


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    iou_sum = 0.0
    count = 0
    with torch.no_grad():
        for grids, _ in loader:
            grids = grids.to(device)
            logits, _ = model(grids)
            total_loss += float(loss_fn(logits, grids)) * grids.shape[0]
            iou_sum += voxel_iou(occupancy_from_logits(logits), grids) * grids.shape[0]
            count += grids.shape[0]
    return total_loss / count, iou_sum / count


def _save_recon_preview(
    model: nn.Module,
    dataset: OccupancyDataset,
    output_path: Path,
    preview_count: int,
    device: torch.device,
) -> None:
    model.eval()
    entries = []
    with torch.no_grad():
        for index in range(preview_count):
            grid, _ = dataset[index]
            logits, _ = model(grid.unsqueeze(0).to(device))
            recon = occupancy_from_logits(logits)[0, 0].cpu().numpy()
            entries.append((grid[0].numpy(), recon, dataset.records[index]))
    save_recon_grid(entries, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny 3D autoencoder.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/autoencoder.json"),
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        help="Overfit a tiny subset of N train examples as a sanity check.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        config.update(read_json(config_path))
    return config


def _make_run_dir(output_root: Path, experiment: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-{experiment}")


if __name__ == "__main__":
    main()
