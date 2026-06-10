from __future__ import annotations

from pathlib import Path

import torch

from tiny3dlatent.data.generate import generate_dataset
from tiny3dlatent.models.autoencoder import Autoencoder
from tiny3dlatent.models.dataset import OccupancyDataset
from tiny3dlatent.models.metrics import voxel_iou
from tiny3dlatent.models.train_ae import train_autoencoder


def test_autoencoder_shapes() -> None:
    model = Autoencoder(resolution=16, latent_dim=32, base_channels=8)
    grids = torch.zeros(2, 1, 16, 16, 16)

    logits, latents = model(grids)

    assert logits.shape == (2, 1, 16, 16, 16)
    assert latents.shape == (2, 32)


def test_voxel_iou_known_values() -> None:
    target = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
    target[0, 0, :2] = True

    identical = voxel_iou(target, target)
    disjoint = torch.zeros_like(target)
    disjoint[0, 0, 2:] = True
    half = torch.zeros_like(target)
    half[0, 0, :1] = True

    assert identical == 1.0
    assert voxel_iou(disjoint, target) == 0.0
    assert voxel_iou(half, target) == 0.5
    assert voxel_iou(torch.zeros_like(target), torch.zeros_like(target)) == 1.0


def test_occupancy_dataset_loads_split(tmp_path: Path) -> None:
    dataset_dir = _make_tiny_dataset(tmp_path)

    train_set = OccupancyDataset(dataset_dir, split="train")
    val_set = OccupancyDataset(dataset_dir, split="val")
    grid, shape_index = train_set[0]

    assert len(train_set) == 8
    assert len(val_set) == 4
    assert grid.shape == (1, 16, 16, 16)
    assert grid.dtype == torch.float32
    assert 0 <= shape_index < 6


def test_train_autoencoder_overfits_tiny_dataset(tmp_path: Path) -> None:
    dataset_dir = _make_tiny_dataset(tmp_path)

    result = train_autoencoder(
        {
            "seed": 0,
            "dataset_dir": dataset_dir,
            "output_dir": (tmp_path / "runs").as_posix(),
            "device": "cpu",
            "resolution": 16,
            "latent_dim": 32,
            "base_channels": 8,
            "batch_size": 8,
            "learning_rate": 3e-3,
            "pos_weight": 3.0,
            "epochs": 80,
            "overfit_count": 8,
            "preview_count": 4,
        }
    )

    run_dir = Path(result["run_dir"])
    assert (run_dir / "config.json").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "autoencoder.pt").exists()
    assert (run_dir / "recon_grid.png").exists()
    assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]
    assert result["best_val_iou"] > 0.9


def _make_tiny_dataset(tmp_path: Path) -> str:
    result = generate_dataset(
        {
            "seed": 3,
            "resolution": 16,
            "train_count": 8,
            "val_count": 4,
            "dataset_dir": (tmp_path / "procedural").as_posix(),
            "output_dir": (tmp_path / "dataset-runs").as_posix(),
            "preview_count": 1,
        }
    )
    return result["dataset_dir"]
