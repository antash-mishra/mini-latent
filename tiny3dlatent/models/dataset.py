from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny3dlatent.data.labels import COLOR_RGB, DESCRIPTOR_MATERIAL, SHAPE_TYPES
from tiny3dlatent.utils.io import read_json

SHAPE_TYPE_TO_INDEX = {name: index for index, name in enumerate(SHAPE_TYPES)}


class OccupancyDataset(Dataset):
    """Occupancy grids from the procedural dataset as `(1, R, R, R)` float tensors.

    Each item is `(occupancy, shape_type_index)`; the index follows the order of
    `tiny3dlatent.data.labels.SHAPE_TYPES` and is used for class conditioning later.
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        *,
        split: str,
        limit: int | None = None,
    ) -> None:
        metadata = read_json(Path(dataset_dir) / "metadata.json")
        records = [r for r in metadata["records"] if r["split"] == split]
        if limit is not None:
            records = records[:limit]
        if not records:
            raise ValueError(f"no records found for split {split!r}")
        self.records: list[dict[str, Any]] = records
        self.resolution = int(metadata["resolution"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        occupancy = np.load(record["grid_file"]).astype(np.float32)
        tensor = torch.from_numpy(occupancy).unsqueeze(0)
        return tensor, SHAPE_TYPE_TO_INDEX[str(record["shape_type"])]


class ColoredOccupancyDataset(OccupancyDataset):
    """Occupancy plus uniform RGB and material channels, with PBR targets.

    Items are `(grid, material)` where `grid` is `(6, R, R, R)` — occupancy,
    RGB in [0, 1], then roughness and metallic, all painted only inside the
    shape — and `material` is `(roughness, metallic)` from the descriptor
    label. Material must be an *input* channel: it affects neither geometry
    nor color, so a latent encoded from those alone could never predict it.
    """

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        record = self.records[index]
        occupancy = np.load(record["grid_file"]).astype(np.float32)
        rgb = np.array(COLOR_RGB[str(record["color"])], dtype=np.float32) / 255.0
        material_spec = DESCRIPTOR_MATERIAL[str(record["descriptor"])]
        per_voxel = np.array(
            [*rgb, material_spec["roughness"], material_spec["metallic"]],
            dtype=np.float32,
        )
        painted = occupancy[None, :, :, :] * per_voxel[:, None, None, None]
        grid = np.concatenate([occupancy[None], painted], axis=0)

        material = torch.tensor(
            [material_spec["roughness"], material_spec["metallic"]],
            dtype=torch.float32,
        )
        return torch.from_numpy(grid), material
