from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from tiny3dlatent.models.common import count_parameters, select_device, set_seed
from tiny3dlatent.models.dataset import OccupancyDataset
from tiny3dlatent.models.vae import VAE, kl_per_dimension
from tiny3dlatent.utils.io import ensure_dir, read_json, write_json

DEFAULT_CONFIG = {
    "seed": 0,
    "output_dir": "outputs/runs",
    "device": "auto",
    "latent_dim": 128,
    "base_channels": 16,
    "warmup_steps": 3,
    "timed_steps": 15,
    "settings": [
        {
            "label": "32^3",
            "dataset_dir": "data/procedural",
            "resolution": 32,
            "batch_size": 32,
        },
        {
            "label": "64^3",
            "dataset_dir": "data/procedural64",
            "resolution": 64,
            "batch_size": 16,
        },
    ],
}


def main() -> None:
    args = _parse_args()
    config = DEFAULT_CONFIG.copy()
    if args.config and args.config.exists():
        config.update(read_json(args.config))
    benchmark(config)


def benchmark(config: dict[str, Any]) -> dict[str, Any]:
    device = select_device(str(config["device"]))
    run_dir = _make_run_dir(Path(str(config["output_dir"])))

    rows = []
    for setting in config["settings"]:
        rows.append(_measure(setting, config, device))
        print(
            f"{rows[-1]['label']:>6}: {rows[-1]['seconds_per_step']:.3f}s/step "
            f"(batch {rows[-1]['batch_size']}), "
            f"{rows[-1]['seconds_per_example'] * 1000:.1f}ms/example, "
            f"params {rows[-1]['parameter_count'] / 1e6:.1f}M, "
            f"device_mem {rows[-1]['device_memory_mb']:.0f}MB"
        )

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "warmup_steps": config["warmup_steps"],
        "timed_steps": config["timed_steps"],
        "rows": rows,
        "run_dir": run_dir.as_posix(),
    }
    write_json(run_dir / "benchmark.json", result)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metadata.json", result)
    print(f"benchmark -> {run_dir / 'benchmark.json'}")
    return result


def _measure(
    setting: dict[str, Any], config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    dataset = OccupancyDataset(Path(str(setting["dataset_dir"])), split="train")
    loader = DataLoader(dataset, batch_size=int(setting["batch_size"]), shuffle=True)
    model = VAE(
        resolution=int(setting["resolution"]),
        latent_dim=int(config["latent_dim"]),
        base_channels=int(config["base_channels"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    warmup = int(config["warmup_steps"])
    timed = int(config["timed_steps"])
    step_index = 0
    started = None
    while step_index < warmup + timed:
        for grids, _ in loader:
            if step_index == warmup:
                _synchronize(device)
                started = time.time()
            grids = grids.to(device)
            logits, mean, logvar, _ = model(grids)
            loss = loss_fn(logits, grids) + 1e-4 * kl_per_dimension(mean, logvar).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_index += 1
            if step_index >= warmup + timed:
                break
    _synchronize(device)
    assert started is not None
    elapsed = time.time() - started

    seconds_per_step = elapsed / timed
    return {
        "label": str(setting["label"]),
        "resolution": int(setting["resolution"]),
        "batch_size": int(setting["batch_size"]),
        "voxels_per_example": int(setting["resolution"]) ** 3,
        "parameter_count": count_parameters(model),
        "seconds_per_step": seconds_per_step,
        "seconds_per_example": seconds_per_step / int(setting["batch_size"]),
        "device_memory_mb": _device_memory_mb(device),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _device_memory_mb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure VAE training cost at 32^3 vs 64^3."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark_resolution.json"),
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(output_root / f"{timestamp}-resolution-benchmark")


if __name__ == "__main__":
    main()
