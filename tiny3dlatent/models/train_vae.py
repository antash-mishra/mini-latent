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

from tiny3dlatent.models.common import (
    count_parameters,
    save_checkpoint,
    select_device,
    set_seed,
)
from tiny3dlatent.models.dataset import OccupancyDataset
from tiny3dlatent.models.metrics import occupancy_from_logits, voxel_iou
from tiny3dlatent.models.recon_preview import save_recon_grid
from tiny3dlatent.models.vae import VAE, kl_per_dimension
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
    "kl_weight": 1e-4,
    "kl_warmup_epochs": 10,
    "epochs": 40,
    "preview_count": 8,
}

ACTIVE_DIM_THRESHOLD = 0.01


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train_vae(config)


def train_vae(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))

    dataset_dir = Path(str(config["dataset_dir"]))
    train_set = OccupancyDataset(dataset_dir, split="train")
    val_set = OccupancyDataset(dataset_dir, split="val")
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = VAE(
        resolution=int(config["resolution"]),
        latent_dim=int(config["latent_dim"]),
        base_channels=int(config["base_channels"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    pos_weight = torch.tensor(float(config["pos_weight"]), device=device)
    recon_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    kl_weight = float(config["kl_weight"])
    warmup_epochs = int(config["kl_warmup_epochs"])
    epochs = int(config["epochs"])

    run_dir = _make_run_dir(Path(str(config["output_dir"])))
    started = time.time()

    history: list[dict[str, float]] = []
    best_val_iou = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        beta = kl_weight * min(1.0, epoch / max(1, warmup_epochs))
        model.train()
        recon_sum = 0.0
        kl_sum = 0.0
        for grids, _ in train_loader:
            grids = grids.to(device)
            logits, mean, logvar, _ = model(grids)
            recon_loss = recon_loss_fn(logits, grids)
            kl_total = kl_per_dimension(mean, logvar).sum()
            loss = recon_loss + beta * kl_total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recon_sum += float(recon_loss.detach()) * grids.shape[0]
            kl_sum += float(kl_total.detach()) * grids.shape[0]
        train_recon = recon_sum / len(train_set)
        train_kl = kl_sum / len(train_set)

        val_iou_mean, val_iou_sampled, val_kl_per_dim = _evaluate(
            model, val_loader, device
        )
        history.append(
            {
                "epoch": epoch,
                "beta": beta,
                "train_recon_loss": train_recon,
                "train_kl": train_kl,
                "val_iou_mean_latent": val_iou_mean,
                "val_iou_sampled_latent": val_iou_sampled,
                "val_kl_total": float(val_kl_per_dim.sum()),
                "val_active_dims": int((val_kl_per_dim > ACTIVE_DIM_THRESHOLD).sum()),
            }
        )
        if val_iou_mean > best_val_iou:
            best_val_iou = val_iou_mean
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % max(1, epochs // 20) == 0 or epoch == epochs:
            print(
                f"epoch {epoch:4d}/{epochs}  recon {train_recon:.4f}  "
                f"kl {train_kl:.1f}  beta {beta:.2e}  "
                f"val_iou {val_iou_mean:.4f}  sampled {val_iou_sampled:.4f}  "
                f"active_dims {history[-1]['val_active_dims']}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(
        run_dir / "vae.pt",
        model,
        config=config,
        epoch=best_epoch,
        val_iou=best_val_iou,
    )

    val_iou_mean, val_iou_sampled, val_kl_per_dim = _evaluate(model, val_loader, device)
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
        "best_epoch": best_epoch,
        "best_val_iou_mean_latent": best_val_iou,
        "final_val_iou_sampled_latent": val_iou_sampled,
        "final_val_kl_total": float(val_kl_per_dim.sum()),
        "final_val_active_dims": int((val_kl_per_dim > ACTIVE_DIM_THRESHOLD).sum()),
        "kl_per_dim_max": float(val_kl_per_dim.max()),
        "kl_per_dim_mean": float(val_kl_per_dim.mean()),
        "runtime_seconds": runtime,
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metadata.json", metadata)

    print(
        f"best val_iou {best_val_iou:.4f} (epoch {best_epoch})  "
        f"sampled {val_iou_sampled:.4f}  kl_total {float(val_kl_per_dim.sum()):.1f}  "
        f"active_dims {metadata['final_val_active_dims']} -> {run_dir}"
    )
    return {
        "run_dir": run_dir.as_posix(),
        "best_val_iou": best_val_iou,
        "val_iou_sampled": val_iou_sampled,
        "val_kl_total": float(val_kl_per_dim.sum()),
        "val_active_dims": metadata["final_val_active_dims"],
        "history": history,
    }


def _evaluate(
    model: VAE, loader: DataLoader, device: torch.device
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    iou_mean_sum = 0.0
    iou_sampled_sum = 0.0
    kl_sum: torch.Tensor | None = None
    count = 0
    with torch.no_grad():
        for grids, _ in loader:
            grids = grids.to(device)
            mean, logvar = model.encode(grids)
            logits_mean = model.decode(mean)
            logits_sampled = model.decode(model.reparameterize(mean, logvar))
            iou_mean_sum += (
                voxel_iou(occupancy_from_logits(logits_mean), grids) * grids.shape[0]
            )
            iou_sampled_sum += (
                voxel_iou(occupancy_from_logits(logits_sampled), grids) * grids.shape[0]
            )
            batch_kl = kl_per_dimension(mean, logvar) * grids.shape[0]
            kl_sum = batch_kl if kl_sum is None else kl_sum + batch_kl
            count += grids.shape[0]
    assert kl_sum is not None
    return iou_mean_sum / count, iou_sampled_sum / count, (kl_sum / count).cpu()


def _save_recon_preview(
    model: VAE,
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
            mean, _ = model.encode(grid.unsqueeze(0).to(device))
            recon = occupancy_from_logits(model.decode(mean))[0, 0].cpu().numpy()
            entries.append((grid[0].numpy(), recon, dataset.records[index]))
    save_recon_grid(entries, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny 3D VAE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae.json"),
        help="Path to a JSON config file.",
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


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-vae")


if __name__ == "__main__":
    main()
