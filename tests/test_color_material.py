from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from tiny3dlatent.data.generate import generate_dataset
from tiny3dlatent.data.labels import COLOR_RGB, DESCRIPTOR_MATERIAL
from tiny3dlatent.models.color_vae import ColorVAE, masked_color_loss
from tiny3dlatent.models.dataset import ColoredOccupancyDataset
from tiny3dlatent.representation.export import (
    export_glb_with_material,
    load_glb_material,
)


def test_colored_dataset_channels_match_palette(tmp_path: Path) -> None:
    dataset_dir = _make_tiny_dataset(tmp_path)

    dataset = ColoredOccupancyDataset(dataset_dir, split="train")
    grid, material = dataset[0]
    record = dataset.records[0]
    expected_rgb = np.array(COLOR_RGB[str(record["color"])]) / 255.0
    expected_material = DESCRIPTOR_MATERIAL[str(record["descriptor"])]
    expected_channels = [
        *expected_rgb,
        expected_material["roughness"],
        expected_material["metallic"],
    ]

    occupancy = grid[0]
    assert grid.shape == (6, 16, 16, 16)
    assert material.shape == (2,)
    assert abs(float(material[0]) - expected_material["roughness"]) < 1e-6
    assert abs(float(material[1]) - expected_material["metallic"]) < 1e-6
    mask = occupancy.bool()
    for channel, expected in enumerate(expected_channels):
        values = grid[1 + channel][mask]
        assert torch.allclose(values, torch.full_like(values, float(expected)))
        assert grid[1 + channel][~mask].abs().max() == 0.0


def test_color_vae_shapes() -> None:
    model = ColorVAE(resolution=16, latent_dim=32, base_channels=8)
    grids = torch.rand(2, 6, 16, 16, 16)

    logits, rgb, material, mean, logvar = model(grids)

    assert logits.shape == (2, 1, 16, 16, 16)
    assert rgb.shape == (2, 3, 16, 16, 16)
    assert material.shape == (2, 2)
    assert mean.shape == (2, 32)
    assert logvar.shape == (2, 32)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert material.min() >= 0.0 and material.max() <= 1.0


def test_masked_color_loss_ignores_empty_voxels() -> None:
    occupancy = torch.zeros(1, 1, 4, 4, 4)
    occupancy[0, 0, :2] = 1.0
    target = torch.zeros(1, 3, 4, 4, 4)
    predicted = torch.ones(1, 3, 4, 4, 4)
    # Perfect inside the shape, wrong outside: loss must be zero.
    predicted_inside_only = target.clone()
    predicted_inside_only[0, :, 2:] = 5.0

    assert float(masked_color_loss(predicted_inside_only, target, occupancy)) == 0.0
    assert float(masked_color_loss(predicted, target, occupancy)) == 1.0


def test_glb_material_roundtrip(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    palette_red = tuple(value / 255.0 for value in COLOR_RGB["red"])
    path = tmp_path / "asset.glb"

    export_glb_with_material(
        mesh, path, base_color=palette_red, roughness=0.3, metallic=1.0, name="red"
    )
    material = load_glb_material(path)

    assert np.allclose(material["base_color"], palette_red, atol=0.01)
    assert material["roughness"] == 0.3
    assert material["metallic"] == 1.0


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
