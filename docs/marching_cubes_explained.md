# Marching Cubes Explained For Tiny3D-Latent

## The short version

Marching cubes converts a **3D scalar grid** into a **triangle mesh**.

In this project, the scalar grid is currently a binary occupancy grid:

```text
0 = empty space
1 = inside the object
```

Marching cubes walks through the grid one small cube at a time and asks:

```text
Which corners of this cube are inside the object, and which are outside?
```

Where the answer changes from outside to inside, the surface must pass through that cube. The
algorithm places triangles there.

That makes marching cubes the bridge between the representation you already built in Milestone 1
and the asset you want later:

```text
occupancy grid -> triangle mesh -> OBJ/GLB -> preview/render/export
```

## Why Milestone 2 needs it

Right now the dataset contains voxel grids such as:

```text
shape_000001_occupancy.npy
```

Those tensors are useful for neural networks, but they are not yet a normal 3D asset. A game engine,
Blender, or a human inspecting the result wants triangles, vertices, and faces.

Milestone 2 proves that the representation can survive this full path:

```text
procedural shape
  -> occupancy grid
  -> marching cubes
  -> cleaned mesh
  -> exported OBJ
```

If that path is wrong, later model outputs will be hard to debug because you will not know whether
the problem comes from the neural network or from the export pipeline.

## A 2D analogy first

It is easier to understand the idea in 2D.

Imagine a grid of pixels storing `0` outside and `1` inside:

```text
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
```

The boundary of the object lies between `0` and `1`. A 2D algorithm called **marching squares**
looks at each little square of four pixels and draws line segments where that boundary crosses.

Marching cubes is the same idea in 3D:

```text
marching squares: 4 corners  -> line segments
marching cubes:   8 corners  -> triangles
```

## What one cube looks like

Every small cube has 8 corner samples:

```text
top face:     4 samples
bottom face:  4 samples
total:        8 samples
```

Each corner is classified relative to an iso-value:

```text
value >= iso_value -> inside
value <  iso_value -> outside
```

For occupancy grids containing only `0` and `1`, `iso_value = 0.5` is the natural boundary:

```text
0 --- surface at 0.5 --- 1
```

If all 8 corners are outside, there is no surface in that cube.

If all 8 corners are inside, there is also no surface in that cube.

Only cubes containing a mix of inside and outside corners contribute triangles.

## Why there are lookup tables

Each of the 8 cube corners can be inside or outside:

```text
2^8 = 256 possible corner patterns
```

Many patterns are rotations or mirror versions of others, but the implementation still uses a
precomputed lookup table to answer:

```text
For this inside/outside pattern, which edges are crossed and which triangles should be emitted?
```

That lookup-table detail is why we use a trusted library implementation rather than hand-coding the
full algorithm in this milestone.

## What marching cubes returns

`skimage.measure.marching_cubes` returns:

```python
vertices, faces, normals, values
```

In plain English:

- `vertices`: points in 3D space
- `faces`: triples of vertex indices, each triple forming one triangle
- `normals`: surface directions used for lighting
- `values`: scalar values near the generated surface

Those arrays can be wrapped in a `trimesh.Trimesh` object and exported as a normal mesh file.

## The important project-specific detail: coordinate spaces

Your dataset code creates coordinates in normalized world space:

```python
lin = np.linspace(-1.0, 1.0, resolution)
```

So conceptually, your shapes live inside:

```text
[-1, 1] x [-1, 1] x [-1, 1]
```

But marching cubes operates on arrays. By default, it returns vertices in **array-index space**:

```text
0, 1, 2, ..., resolution - 1
```

Therefore, Milestone 2 must convert the mesh back into the normalized space used by the rest of the
project.

For a `32^3` grid:

```python
voxel_size = 2.0 / (32 - 1)
world_vertex = -1.0 + index_vertex * voxel_size
```

Because the Milestone 2 plan pads the grid with a one-voxel empty border before extraction, the
actual mapping becomes:

```python
world_vertex = -1.0 + (padded_vertex - 1.0) * voxel_size
```

That `-1.0` offset for the padding is easy to miss, and it is one of the most important details in
the implementation.

## Why padding is necessary

If a shape touches the boundary of the array, marching cubes may see occupied voxels at the edge but
no explicit empty voxels outside them. That can produce an open mesh because the algorithm has no
outside samples to close the surface against.

Padding adds one layer of empty space around the grid:

```text
before:  object may touch the array border
after:   empty shell surrounds the object
```

This lets the surface close properly even when an object reaches the original boundary.

In this project, padding should happen before extraction:

```python
padded = np.pad(occupancy, pad_width=1, mode="constant", constant_values=0)
```

## What marching cubes does not do

Marching cubes is powerful, but it does not magically recover detail that the grid never stored.

At `32^3` resolution:

- round objects can still look faceted
- thin parts may disappear
- sharp edges can become stair-stepped
- a torus with a very thin ring can lose quality

Increasing the grid to `64^3` gives more detail, but costs roughly 8 times more voxels:

```text
32^3 = 32,768 voxels
64^3 = 262,144 voxels
```

That is why the overall plan starts with `32^3`: it is cheap enough to iterate quickly while still
being good enough to validate the pipeline.

## Where marching cubes fits in the whole project

For now:

```text
procedural formula -> occupancy grid -> marching cubes -> mesh
```

Later:

```text
decoder output -> predicted occupancy or TSDF grid -> marching cubes -> generated mesh
```

The mesh extractor should not care whether the grid came from a hand-written procedural shape or a
neural network. That separation is exactly what makes Milestone 2 valuable.

## A useful mental model

Think of marching cubes as a translator:

```text
grid language:  "inside / outside at many sampled locations"
mesh language:  "surface triangles between those regions"
```

It is not the model. It is the final geometric interpreter that turns a field into an inspectable
surface.

