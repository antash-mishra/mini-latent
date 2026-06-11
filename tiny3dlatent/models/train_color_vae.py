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

from tiny3dlatent.models.color_vae import ColorVAE, masked_color_loss
from tiny3dlatent.models.common import (
    count_parameters,
    save_checkpoint,
    select_device,
    set_seed,
)
from tiny3dlatent.models.dataset import ColoredOccupancyDataset
from tiny3dlatent.models.metrics import occupancy_from_logits, voxel_iou
from tiny3dlatent.models.vae import kl_per_dimension
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json

DEFAULT_CONFIG = {
    "seed": 0,
    "dataset_dir": "data/procedural",
    "output_dir": "outputs/runs",
    "device": "auto",
    "model_type": "color_vae",
    "resolution": 32,
    "latent_dim": 128,
    "base_channels": 16,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "pos_weight": 3.0,
    "color_weight": 1.0,
    "material_weight": 0.1,
    "kl_weight": 1e-4,
    "kl_warmup_epochs": 10,
    "epochs": 60,
}


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train_color_vae(config)


def train_color_vae(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))

    dataset_dir = Path(str(config["dataset_dir"]))
    train_set = ColoredOccupancyDataset(dataset_dir, split="train")
    val_set = ColoredOccupancyDataset(dataset_dir, split="val")
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = ColorVAE(
        resolution=int(config["resolution"]),
        latent_dim=int(config["latent_dim"]),
        base_channels=int(config["base_channels"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    pos_weight = torch.tensor(float(config["pos_weight"]), device=device)
    occupancy_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    color_weight = float(config["color_weight"])
    material_weight = float(config["material_weight"])
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
        sums = {"occupancy": 0.0, "color": 0.0, "material": 0.0, "kl": 0.0}
        for grids, materials in train_loader:
            grids = grids.to(device)
            materials = materials.to(device)
            occupancy = grids[:, :1]
            target_rgb = grids[:, 1:4]

            logits, rgb, material, mean, logvar = model(grids)
            occupancy_loss = occupancy_loss_fn(logits, occupancy)
            color_loss = masked_color_loss(rgb, target_rgb, occupancy)
            material_loss = torch.nn.functional.mse_loss(material, materials)
            kl_total = kl_per_dimension(mean, logvar).sum()
            loss = (
                occupancy_loss
                + color_weight * color_loss
                + material_weight * material_loss
                + beta * kl_total
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = grids.shape[0]
            sums["occupancy"] += float(occupancy_loss.detach()) * count
            sums["color"] += float(color_loss.detach()) * count
            sums["material"] += float(material_loss.detach()) * count
            sums["kl"] += float(kl_total.detach()) * count
        averages = {key: value / len(train_set) for key, value in sums.items()}

        val_metrics = _evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "beta": beta, **averages, **val_metrics})
        if val_metrics["val_iou"] > best_val_iou:
            best_val_iou = val_metrics["val_iou"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % max(1, epochs // 15) == 0 or epoch == epochs:
            print(
                f"epoch {epoch:4d}/{epochs}  occ {averages['occupancy']:.4f}  "
                f"color {averages['color']:.4f}  mat {averages['material']:.4f}  "
                f"kl {averages['kl']:.1f}  val_iou {val_metrics['val_iou']:.4f}  "
                f"color_mae {val_metrics['val_color_mae']:.4f}  "
                f"mat_mae {val_metrics['val_material_mae']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(
        run_dir / "color_vae.pt",
        model,
        config=config,
        epoch=best_epoch,
        val_iou=best_val_iou,
    )

    final_metrics = _evaluate(model, val_loader, device)
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
        "best_val_iou": best_val_iou,
        **final_metrics,
        "runtime_seconds": runtime,
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metadata.json", metadata)

    print(
        f"best val_iou {best_val_iou:.4f} (epoch {best_epoch})  "
        f"color_mae {final_metrics['val_color_mae']:.4f}  "
        f"material_mae {final_metrics['val_material_mae']:.4f} -> {run_dir}"
    )
    return {
        "run_dir": run_dir.as_posix(),
        "best_val_iou": best_val_iou,
        **final_metrics,
    }


def _evaluate(
    model: ColorVAE, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    iou_sum = 0.0
    color_error_sum = 0.0
    material_error_sum = 0.0
    count = 0
    with torch.no_grad():
        for grids, materials in loader:
            grids = grids.to(device)
            materials = materials.to(device)
            occupancy = grids[:, :1]
            target_rgb = grids[:, 1:4]
            mean, _ = model.encode(grids)
            logits, rgb, material = model.decode_full(mean)

            batch = grids.shape[0]
            iou_sum += voxel_iou(occupancy_from_logits(logits), occupancy) * batch
            mask = occupancy.expand_as(rgb)
            occupied = mask.sum().clamp_min(1.0)
            color_error_sum += (
                float(((rgb - target_rgb).abs() * mask).sum() / occupied) * batch
            )
            material_error_sum += float((material - materials).abs().mean()) * batch
            count += batch
    return {
        "val_iou": iou_sum / count,
        "val_color_mae": color_error_sum / count,
        "val_material_mae": material_error_sum / count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the color/material 3D VAE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/color_vae.json"),
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
    return ensure_dir(output_root / f"{timestamp}-color-vae")


if __name__ == "__main__":
    main()
