from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from tiny3dlatent.models.common import (
    build_vae_from_checkpoint,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
)
from tiny3dlatent.models.dataset import ColoredOccupancyDataset, OccupancyDataset
from tiny3dlatent.models.flow import ConditionedLatentFlow, rectified_flow_pair
from tiny3dlatent.text.parser import (
    ATTRIBUTE_ORDER,
    ATTRIBUTE_SIZES,
    ATTRIBUTE_VOCABULARIES,
)
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json

DEFAULT_CONFIG = {
    "seed": 0,
    "dataset_dir": "data/procedural",
    "output_dir": "outputs/runs",
    "device": "auto",
    "vae_checkpoint": "latest",
    "attribute_dim": 32,
    "hidden_dim": 256,
    "time_dim": 64,
    "hidden_layers": 3,
    "condition_dropout": 0.1,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 400,
}


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train_text_flow(config)


def train_text_flow(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))

    vae_checkpoint_path = _resolve_vae_checkpoint(
        str(config["vae_checkpoint"]), Path(str(config["output_dir"]))
    )
    vae_checkpoint = load_checkpoint(vae_checkpoint_path)
    vae_config = vae_checkpoint["config"]
    vae = build_vae_from_checkpoint(vae_checkpoint).to(device)

    latents, attributes = _encode_latents_with_attributes(
        vae,
        Path(str(config["dataset_dir"])),
        device,
        colored=vae_config.get("model_type") == "color_vae",
    )
    latent_mean = latents.mean(dim=0)
    latent_std = latents.std(dim=0).clamp_min(1e-5)
    normalized = (latents - latent_mean) / latent_std

    dataset = TensorDataset(normalized, attributes)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True)

    model = ConditionedLatentFlow(
        latent_dim=int(vae_config["latent_dim"]),
        attribute_sizes=ATTRIBUTE_SIZES,
        attribute_dim=int(config["attribute_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        time_dim=int(config["time_dim"]),
        hidden_layers=int(config["hidden_layers"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    condition_dropout = float(config["condition_dropout"])
    null_row = torch.tensor(ATTRIBUTE_SIZES, device=device)

    experiment = (
        "color-text-flow"
        if vae_config.get("model_type") == "color_vae"
        else "text-flow"
    )
    run_dir = _make_run_dir(Path(str(config["output_dir"])), experiment)
    started = time.time()
    epochs = int(config["epochs"])
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_latents, batch_attributes in loader:
            batch_latents = batch_latents.to(device)
            batch_attributes = batch_attributes.to(device)
            if condition_dropout > 0:
                dropped = torch.rand(batch_latents.shape[0], device=device)
                dropped = dropped < condition_dropout
                batch_attributes = torch.where(
                    dropped[:, None], null_row, batch_attributes
                )
            t = torch.rand(batch_latents.shape[0], device=device)
            noise = torch.randn_like(batch_latents)
            z_t, velocity_target = rectified_flow_pair(batch_latents, t, noise)
            velocity = model(z_t, t, batch_attributes)
            loss = torch.nn.functional.mse_loss(velocity, velocity_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * batch_latents.shape[0]
        epoch_loss = loss_sum / len(dataset)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}/{epochs}  flow_loss {epoch_loss:.4f}")

    save_checkpoint(
        run_dir / "text_flow.pt",
        model,
        config=config,
        vae_checkpoint=vae_checkpoint_path.as_posix(),
        latent_mean=latent_mean,
        latent_std=latent_std,
        final_loss=history[-1]["train_loss"],
    )

    runtime = time.time() - started
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "run_dir": run_dir.as_posix(),
        "device": str(device),
        "vae_checkpoint": vae_checkpoint_path.as_posix(),
        "parameter_count": count_parameters(model),
        "train_latents": int(latents.shape[0]),
        "final_loss": history[-1]["train_loss"],
        "runtime_seconds": runtime,
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metadata.json", metadata)

    print(
        f"final flow_loss {history[-1]['train_loss']:.4f} in {runtime:.1f}s -> {run_dir}"
    )
    return {
        "run_dir": run_dir.as_posix(),
        "final_loss": history[-1]["train_loss"],
        "history": history,
    }


def _encode_latents_with_attributes(
    vae: torch.nn.Module,
    dataset_dir: Path,
    device: torch.device,
    *,
    colored: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_class = ColoredOccupancyDataset if colored else OccupancyDataset
    train_set = dataset_class(dataset_dir, split="train")
    loader = DataLoader(train_set, batch_size=64)
    all_means = []
    with torch.no_grad():
        for grids, _ in loader:
            mean, _ = vae.encode(grids.to(device))
            all_means.append(mean.cpu())

    attribute_rows = []
    for record in train_set.records:
        row = [
            ATTRIBUTE_VOCABULARIES[attribute].index(str(record[attribute]))
            for attribute in ATTRIBUTE_ORDER
        ]
        attribute_rows.append(row)
    return torch.cat(all_means), torch.tensor(attribute_rows, dtype=torch.long)


def _resolve_vae_checkpoint(setting: str, output_root: Path) -> Path:
    if setting == "latest":
        pattern, hint = "*-vae/vae.pt", "tiny3dlatent.models.train_vae"
    elif setting == "latest-color":
        pattern, hint = (
            "*-color-vae/color_vae.pt",
            "tiny3dlatent.models.train_color_vae",
        )
    else:
        return Path(setting)
    candidates = sorted(output_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint matching {pattern} under {output_root}; train one with "
            f"python -m {hint}"
        )
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the prompt-attribute-conditioned latent rectified flow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/text_flow.json"),
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


def _make_run_dir(output_root: Path, experiment: str = "text-flow") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-{experiment}")


if __name__ == "__main__":
    main()
