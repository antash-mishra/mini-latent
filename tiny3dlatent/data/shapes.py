from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tiny3dlatent.data.grid import occupancy_to_uint8
from tiny3dlatent.data.labels import COLORS, DESCRIPTORS, SHAPE_TYPES, SIZES, label_for

ShapeType = Literal["sphere", "cube", "rounded_box", "cylinder", "capsule", "torus"]


@dataclass(frozen=True)
class ShapeSpec:
    shape_type: ShapeType
    color: str
    size: str
    descriptor: str
    center: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotation: tuple[float, float, float]

    @property
    def label(self) -> str:
        return label_for(self.color, self.size, self.descriptor, self.shape_type)

    def to_metadata(self) -> dict[str, object]:
        return {
            "shape_type": self.shape_type,
            "color": self.color,
            "size": self.size,
            "descriptor": self.descriptor,
            "label": self.label,
            "parameters": {
                "center": list(self.center),
                "scale": list(self.scale),
                "rotation": list(self.rotation),
            },
        }


def sample_shape_spec(rng: np.random.Generator) -> ShapeSpec:
    shape_type = str(rng.choice(SHAPE_TYPES))
    color = str(rng.choice(COLORS))
    size = str(rng.choice(SIZES))
    descriptor = str(rng.choice(DESCRIPTORS))
    if shape_type == "sphere" and descriptor in ("tall", "wide"):
        descriptor = "standard"

    base_scale = {
        "small": 0.36,
        "medium": 0.48,
        "large": 0.60,
    }[size]

    scale = np.array([base_scale, base_scale, base_scale], dtype=np.float32)
    if descriptor == "tall":
        scale *= np.array([0.82, 1.25, 0.82], dtype=np.float32)
    elif descriptor == "wide":
        scale *= np.array([1.25, 0.82, 1.25], dtype=np.float32)

    if shape_type == "torus":
        scale *= 0.88
    elif shape_type in {"cylinder", "capsule"}:
        scale *= np.array([0.82, 1.12, 0.82], dtype=np.float32)

    center = rng.uniform(-0.12, 0.12, size=3).astype(np.float32)
    rotation = rng.uniform(-0.65, 0.65, size=3).astype(np.float32)
    if shape_type == "sphere":
        rotation[:] = 0.0

    return ShapeSpec(
        shape_type=shape_type,  # type: ignore[arg-type]
        color=color,
        size=size,
        descriptor=descriptor,
        center=tuple(float(v) for v in center),
        scale=tuple(float(v) for v in scale),
        rotation=tuple(float(v) for v in rotation),
    )


def generate_occupancy(
    spec: ShapeSpec,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    x, y, z = _to_local_coordinates(spec, grid)

    if spec.shape_type == "sphere":
        occupancy = _sphere(x, y, z)
    elif spec.shape_type == "cube":
        occupancy = _box(x, y, z)
    elif spec.shape_type == "rounded_box":
        occupancy = _rounded_box(x, y, z)
    elif spec.shape_type == "cylinder":
        occupancy = _cylinder(x, y, z)
    elif spec.shape_type == "capsule":
        occupancy = _capsule(x, y, z)
    elif spec.shape_type == "torus":
        occupancy = _torus(x, y, z)
    else:
        raise ValueError(f"unsupported shape type: {spec.shape_type}")

    return occupancy_to_uint8(occupancy)


def _to_local_coordinates(
    spec: ShapeSpec,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = grid
    centered = np.stack(
        [
            (x - spec.center[0]).ravel(),
            (y - spec.center[1]).ravel(),
            (z - spec.center[2]).ravel(),
        ],
        axis=0,
    )
    rotation = _rotation_matrix(*spec.rotation)
    local = rotation.T @ centered
    shape = x.shape

    sx, sy, sz = spec.scale
    return (
        (local[0].reshape(shape) / sx).astype(np.float32),
        (local[1].reshape(shape) / sy).astype(np.float32),
        (local[2].reshape(shape) / sz).astype(np.float32),
    )


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (x * x + y * y + z * z) <= 1.0


def _box(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0) & (np.abs(z) <= 1.0)


def _rounded_box(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    radius = 0.28
    qx = np.abs(x) - (1.0 - radius)
    qy = np.abs(y) - (1.0 - radius)
    qz = np.abs(z) - (1.0 - radius)
    outside = np.sqrt(
        np.maximum(qx, 0.0) ** 2
        + np.maximum(qy, 0.0) ** 2
        + np.maximum(qz, 0.0) ** 2
    )
    inside = np.minimum(np.maximum.reduce([qx, qy, qz]), 0.0)
    return (outside + inside - radius) <= 0.0


def _cylinder(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    radial = x * x + z * z
    return (radial <= 1.0) & (np.abs(y) <= 1.0)


def _capsule(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    half_height = 0.68
    capped_y = np.clip(y, -half_height, half_height)
    dy = y - capped_y
    return (x * x + dy * dy + z * z) <= 0.52**2


def _torus(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    major_radius = 0.68
    minor_radius = 0.25
    radial = np.sqrt(x * x + z * z)
    return ((radial - major_radius) ** 2 + y * y) <= minor_radius**2
