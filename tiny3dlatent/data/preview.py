from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tiny3dlatent.data.labels import COLOR_RGB
from tiny3dlatent.utils.io import ensure_dir


def save_preview_grid(
    examples: list[tuple[np.ndarray, dict[str, object]]],
    output_path: Path,
    *,
    columns: int = 5,
) -> None:
    if not examples:
        raise ValueError("at least one example is required for preview")

    ensure_dir(output_path.parent)
    rows = int(np.ceil(len(examples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 2.4))
    flat_axes = np.atleast_1d(axes).ravel()

    for axis in flat_axes:
        axis.axis("off")

    for axis, (grid, metadata) in zip(flat_axes, examples, strict=False):
        middle_slice = grid[:, grid.shape[1] // 2, :]
        label = str(metadata["label"])
        color = str(metadata["color"])
        axis.imshow(_colorize_slice(middle_slice, color), interpolation="nearest")
        axis.set_title(label, fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_example_slices(
    grid: np.ndarray,
    output_path: Path,
    *,
    title: str,
    color: str,
) -> None:
    ensure_dir(output_path.parent)
    slices = [
        ("x middle", grid[grid.shape[0] // 2, :, :]),
        ("y middle", grid[:, grid.shape[1] // 2, :]),
        ("z middle", grid[:, :, grid.shape[2] // 2]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for axis, (name, image) in zip(axes, slices, strict=True):
        axis.imshow(_colorize_slice(image, color), interpolation="nearest")
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _colorize_slice(slice_2d: np.ndarray, color: str) -> np.ndarray:
    rgb = np.array(COLOR_RGB[color], dtype=np.uint8)
    image = np.full((*slice_2d.shape, 3), 238, dtype=np.uint8)
    image[slice_2d.astype(bool)] = rgb
    return image
