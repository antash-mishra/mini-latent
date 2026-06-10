from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tiny3dlatent.data.labels import COLOR_RGB
from tiny3dlatent.utils.io import ensure_dir


def save_recon_grid(
    entries: list[tuple[np.ndarray, np.ndarray, dict[str, object]]],
    output_path: Path,
) -> None:
    """Original-vs-reconstruction middle slices: one column per example.

    Each entry is `(original, reconstruction, record)` with binary `R^3` grids.
    """
    if not entries:
        raise ValueError("at least one entry is required for the recon grid")

    ensure_dir(output_path.parent)
    columns = len(entries)
    figure, axes = plt.subplots(2, columns, figsize=(columns * 2.0, 4.6))
    axes = np.atleast_2d(axes)
    if axes.shape != (2, columns):
        axes = axes.reshape(2, columns)

    for column, (original, reconstruction, record) in enumerate(entries):
        color = str(record["color"])
        axes[0, column].imshow(
            _colorize_slice(original[:, original.shape[1] // 2, :], color),
            interpolation="nearest",
        )
        axes[0, column].set_title(str(record["label"]), fontsize=7)
        axes[1, column].imshow(
            _colorize_slice(reconstruction[:, reconstruction.shape[1] // 2, :], color),
            interpolation="nearest",
        )
        for row in (0, 1):
            axes[row, column].axis("off")

    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("reconstruction", fontsize=9)
    figure.suptitle("top: original / bottom: reconstruction", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _colorize_slice(slice_2d: np.ndarray, color: str) -> np.ndarray:
    rgb = np.array(COLOR_RGB[color], dtype=np.uint8)
    image = np.full((*slice_2d.shape, 3), 238, dtype=np.uint8)
    image[slice_2d.astype(bool)] = rgb
    return image
