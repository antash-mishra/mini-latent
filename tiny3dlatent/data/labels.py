from __future__ import annotations

SHAPE_TYPES = ("sphere", "cube", "rounded_box", "cylinder", "capsule", "torus")
COLORS = ("red", "green", "blue", "yellow", "cyan", "orange")
SIZES = ("small", "medium", "large")
DESCRIPTORS = ("standard", "tall", "wide", "metallic", "matte")


COLOR_RGB = {
    "red": (220, 55, 65),
    "green": (55, 170, 105),
    "blue": (65, 105, 220),
    "yellow": (230, 190, 55),
    "cyan": (55, 190, 210),
    "orange": (230, 130, 55),
}


def label_for(color: str, size: str, descriptor: str, shape_type: str) -> str:
    parts = [color, size]
    if descriptor != "standard":
        parts.append(descriptor)
    parts.append(shape_type.replace("_", " "))
    return " ".join(parts)

