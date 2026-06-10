from __future__ import annotations

import numpy as np


def make_grid(resolution: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create x/y/z coordinate arrays in the normalized range [-1, 1]."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    return np.meshgrid(lin, lin, lin, indexing="ij")


def occupancy_to_uint8(occupancy: np.ndarray) -> np.ndarray:
    """Convert a boolean occupancy mask to compact uint8 storage."""
    return occupancy.astype(np.uint8, copy=False)
