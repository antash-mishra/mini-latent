import numpy as np

# Use a small grid so we can print everything
lin = np.linspace(-1, 1, 3)
print("lin =", lin)
print()

x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")

print("Shapes:", x.shape, y.shape, z.shape)
print()

print("=== X array ===")
print("X[i,j,k] = lin[i] (changes with the FIRST index)")
print(x)
print()

print("=== Y array ===")
print("Y[i,j,k] = lin[j] (changes with the SECOND index)")
print(y)
print()

print("=== Z array ===")
print("Z[i,j,k] = lin[k] (changes with the THIRD index)")
print(z)
print()

# Let's pick a specific point to verify
i, j, k = 1, 0, 2
print(f"At index ({i},{j},{k}):")
print(f"  x[{i},{j},{k}] = {x[i, j, k]}  (should be lin[{i}] = {lin[i]})")
print(f"  y[{i},{j},{k}] = {y[i, j, k]}  (should be lin[{j}] = {lin[j]})")
print(f"  z[{i},{j},{k}] = {z[i, j, k]}  (should be lin[{k}] = {lin[k]})")
print(f"  -> Point: ({x[i, j, k]}, {y[i, j, k]}, {z[i, j, k]})")
print()

# Show a 2D slice (fix i=1)
print("2D slice at i=1 (x = 0.0):")
print("y-z plane coordinates:")
for j in range(3):
    for k in range(3):
        print(f"  ({y[1, j, k]:.1f}, {z[1, j, k]:.1f})", end="  ")
    print()
