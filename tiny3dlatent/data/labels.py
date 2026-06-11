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

# PBR material targets implied by the descriptor labels. Only metallic/matte
# change the material; the geometry descriptors use a neutral default.
DESCRIPTOR_MATERIAL = {
    "metallic": {"roughness": 0.3, "metallic": 1.0},
    "matte": {"roughness": 0.9, "metallic": 0.0},
    "standard": {"roughness": 0.6, "metallic": 0.0},
    "tall": {"roughness": 0.6, "metallic": 0.0},
    "wide": {"roughness": 0.6, "metallic": 0.0},
}


def label_for(color: str, size: str, descriptor: str, shape_type: str) -> str:
    parts = [color, size]
    if descriptor != "standard":
        parts.append(descriptor)
    parts.append(shape_type.replace("_", " "))
    return " ".join(parts)
