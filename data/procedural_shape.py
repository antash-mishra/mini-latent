import numpy as np

SHAPE_TYPES = ["sphere", "cube", "rounded_box", "cylinder", "capsule", "torus"]
COLORS = ["red", "green", "blue", "yellow", "cyan", "orange"]
SIZES = ["small", "medium", "large"]


# This creates coordinates for a 3D grid of points in the range [-1, 1] with a specified resolution
def make_grid(res=32):
    lin = np.linspace(-1, 1, res)
    x, y, z = np.meshgrid(lin, lin, lin)
    return x, y, z

