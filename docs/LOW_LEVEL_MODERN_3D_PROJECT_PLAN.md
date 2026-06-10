# Low-Level Modern 3D Generation Project Plan

Research context checked: 2026-05-12
Plan reviewed against implementation: 2026-06-10

## Current Status

- Milestone 0: folded into Milestone 1. There is no separate smoke script; the dataset generator
  creates timestamped run folders under `outputs/runs/` with config, metadata, and stats, which
  covers the reproducibility goal.
- Milestone 1: complete. `tiny3dlatent/data/` generates 1000 train / 200 val occupancy grids at
  `32^3` with shape/color/size/descriptor labels, train/val split, stats, previews, and tests.
- Milestone 2: complete. `tiny3dlatent/representation/` extracts padded marching cubes
  meshes from occupancy grids, cleans them, exports OBJ, computes mesh stats, and renders
  shaded previews. All six shape types extract as watertight single-component meshes; see
  `tests/test_mesh_extraction.py` and `outputs/runs/<timestamp>-mesh-extraction/`.
- Milestone 3: complete. `tiny3dlatent/models/` trains a 1.3M-parameter 3D conv autoencoder
  (GroupNorm blocks, 128-d latent, BCE with `pos_weight`) via
  `tiny3dlatent.models.train_ae`. Overfit-8 sanity reaches IoU 0.99; the full run reaches
  val IoU 0.87 (> 0.85 target) with all shape types recognizable in `recon_grid.png`; see
  `tests/test_autoencoder.py` and `outputs/runs/20260610-160210-autoencoder/`.
- Milestone 4: complete. `tiny3dlatent/models/vae.py` + `train_vae.py` add mean/log-variance
  heads, reparameterization, and KL warmup to `kl_weight=1e-4`. Val IoU 0.84 from mean
  latents (vs 0.87 AE), 0.81 from sampled latents, KL 117 nats with all 128 dims active (no
  collapse). `interpolate_cli.py` renders smooth cube->cylinder / sphere->cube /
  torus->capsule strips; see `tests/test_vae.py`, `outputs/runs/20260610-161522-vae/`, and
  `outputs/runs/20260610-162631-vae-interpolation/`.

## Short Answer

If you want a low-level project that feels modern but does not cost too much, do this:

```text
Tiny Native 3D Latent Generator
```

Build a miniature version of what current 3D generation systems are doing:

```text
synthetic 3D shapes
  -> voxel / TSDF / compact 3D representation
  -> tiny 3D autoencoder or VAE
  -> tiny latent flow/diffusion model
  -> generated 3D shape
  -> marching cubes mesh export
  -> simple material/color prediction
```

This is low-level because you implement the actual representation, encoder, decoder, generator,
mesh extraction, and evaluation. It is modern because it follows the same broad direction as
TRELLIS, Hunyuan3D, Step1X-3D, and TripoSG: learned 3D representations and latent-space generation.

It is cheap because you can train on procedural toy data instead of downloading or training on
millions of real 3D assets.

## Why This Is Better For Your Goal

Your original TinySDS plan teaches the older DreamFusion path:

```text
text -> 2D diffusion model -> optimize one 3D object slowly
```

That is useful, but current systems are moving more toward:

```text
condition -> learned 3D latent -> decoder -> mesh/material asset
```

So the low-cost modern project should be a toy version of that newer idea:

```text
tiny dataset -> tiny 3D VAE -> tiny latent generator -> mesh
```

You will not beat TRELLIS or Hunyuan3D. That is not the point. The point is to understand the
mechanism at a scale you can afford.

## What You Should Build

Project name:

```text
mini-latent
```

(The Python package is `tiny3dlatent`.)

Final portfolio angle:

```text
I built a small native-3D generative model from scratch: procedural 3D data, voxel/TSDF representation,
3D VAE, latent flow model, mesh extraction, and simple material output.
```

Core pipeline:

```text
prompt label: "red rounded cube"
  -> condition vector
  -> tiny latent generator
  -> 3D latent
  -> decoder
  -> occupancy/TSDF/color grid
  -> marching cubes
  -> GLB/OBJ mesh
```

## Why This Matches Modern Models

| Modern system idea | Tiny version you build |
|---|---|
| TRELLIS.2 uses a compact 3D representation and generative model | Build a tiny voxel/TSDF representation and train a small latent generator |
| Hunyuan3D separates shape and texture | First generate shape, then add color/material channels |
| Step1X-3D uses geometry generation plus texture synthesis | Build geometry first, then a simple color/material decoder |
| TripoSG uses learned geometry representations and flow-style generation | Train a tiny rectified-flow or diffusion model over shape latents |
| Modern systems export assets | Use marching cubes and export OBJ/GLB |

## Cost Control Rules

- Use synthetic procedural data first.
- Start at `32^3` resolution.
- Move to `64^3` only after everything works.
- Use tiny networks, not giant transformers.
- Train on CPU/MPS/small GPU.
- Avoid external paid APIs.
- Avoid real 3D datasets at the start.
- Keep the first dataset under 5,000 generated shapes.

Suggested dependencies:

- PyTorch
- NumPy
- scikit-image for marching cubes
- trimesh for mesh export and inspection
- matplotlib or pyrender/Blender for previews

## Milestones

Each milestone should end with something you can publish: a render, a table, a short explanation,
a failure analysis, or a small demo. The devlog is part of the milestone, not a separate chore.

### Milestone 0: Project Skeleton And Reproducible Runs

Goal:

Make the project easy to run, repeat, and write about.

Build:

- `tiny3dlatent/` Python package.
- `configs/` for dataset, model, and training settings.
- `outputs/runs/<date>-<experiment>/` for artifacts.
- Run metadata with seed, config, runtime, device, git commit, and output paths.
- A tiny smoke script that creates one output folder and writes metadata.

Study:

- Why ML experiments need reproducible run folders.
- What metadata matters when debugging generative models.
- How current 3D projects organize checkpoints, renders, and meshes.

Resources:

- PyTorch quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- PyTorch saving/loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- Hydra config docs, optional if you want a config framework: https://hydra.cc/docs/intro/
- Weights & Biases experiment tracking guide, useful conceptually even if you do not use W&B: https://docs.wandb.ai/guides/track/

Artifact:

```text
python -m tiny3dlatent.experiments.smoke
```

creates a timestamped run folder with `metadata.json`.

Verification:

- Run `./venv/bin/python -m tiny3dlatent.data.generate --config configs/procedural_dataset.json`.
- Check a new `outputs/runs/<timestamp>-procedural-dataset/` folder appears containing
  `metadata.json`, `config.json`, and `dataset_stats.json`.
- File checks are enough here; no screenshot needed.

Devlog angle:

```text
Devlog 0: Why I started with experiment structure before model code
```

Why this milestone exists:

Generative projects create many checkpoints, meshes, previews, and metrics. If the project is not organized from day one, it becomes hard to explain what changed between experiments.

### Milestone 1: Procedural 3D Dataset

Goal:

Generate cheap 3D training data yourself.

Build:

- Occupancy grids for spheres, boxes, rounded boxes, cylinders, capsules, and torus shapes.
- Random scale, position, rotation, and color labels.
- Labels such as `sphere`, `cube`, `cylinder`, `red`, `blue`, `metallic`, `matte`, `tall`, and `wide`.
- Train/validation split.
- Dataset preview script.

Study:

- Occupancy grids: a voxel is either inside or outside the object.
- Signed distance fields: each voxel stores distance to the surface.
- Why synthetic data is useful for low-cost model development.
- How procedural data can teach a model controlled structure before real datasets.

Resources:

- Scratchapixel implicit surfaces and ray concepts: https://www.scratchapixel.com/
- Inigo Quilez distance functions, useful for procedural SDF shapes: https://iquilezles.org/articles/distfunctions/
- NumPy broadcasting basics: https://numpy.org/doc/stable/user/basics.broadcasting.html
- PyTorch Dataset and DataLoader tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- ShapeNet paper, useful context for 3D datasets even if you do not use it: https://arxiv.org/abs/1512.03012

Artifact:

```text
100 generated shapes -> preview grid + dataset_stats.json
```

Verification:

- Run the tests: `./venv/bin/python -m pytest tests/test_procedural_dataset.py`.
- Check `data/procedural/dataset_stats.json`: all six shape types present, split counts match
  the config.
- Visual check: take a screenshot of (or read) `outputs/runs/<run>/preview_grid.png` and
  confirm shapes are recognizable, centered, and not clipped at the grid border.

Devlog angle:

```text
Devlog 1: Building a tiny 3D dataset instead of downloading a giant one
```

Why this milestone exists:

This keeps the project cheap. You get enough structure to train a model while avoiding large datasets, licensing issues, and expensive preprocessing.

### Milestone 2: 3D Representation And Mesh Extraction

Goal:

Choose the 3D format your model will learn and prove it can become a mesh.

Build:

- Dense `32^3` occupancy grid (already produced by Milestone 1).
- Optional TSDF grid after occupancy works.
- Grid visualization slices (already available from Milestone 1 previews).
- Marching cubes mesh extraction.
- Pad the grid with a layer of empty voxels before marching cubes so shapes touching the
  border still produce closed, watertight meshes.
- OBJ or GLB export.
- Simple mesh stats: vertices, faces, bounds, file size, watertightness.

Study:

- Occupancy vs TSDF vs mesh.
- Why neural 3D systems often learn implicit or grid-like representations before exporting meshes.
- Marching cubes as the bridge from grid representation to mesh asset.
- Why resolution matters: `32^3` is cheap but blocky, `64^3` is sharper but more expensive.

Resources:

- scikit-image marching cubes docs: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes
- Marching cubes paper summary/context: https://en.wikipedia.org/wiki/Marching_cubes
- trimesh docs: https://trimesh.org/
- PyVista examples, optional for visualization: https://docs.pyvista.org/examples/
- NVIDIA Kaolin representation docs, useful context: https://kaolin.readthedocs.io/en/latest/modules/kaolin.rep.html

Artifact:

```text
procedural occupancy grid -> mesh.obj / mesh.glb -> turntable or preview images
```

Verification:

- Programmatic: load the exported mesh with trimesh and assert vertices > 0, faces > 0,
  `mesh.is_watertight` is true, and the bounds fit inside the expected box.
- Run a known sphere grid through the pipeline and check the vertex count and bounds are
  plausible for a sphere.
- Visual check: render preview images of extracted meshes and take a screenshot — surfaces
  should be closed, with no holes where shapes touch the grid border.

Devlog angle:

```text
Devlog 2: Turning a voxel grid into my first generated mesh
```

Why this milestone exists:

Modern 3D generation is not only about tensors. The output must eventually become an inspectable asset. This milestone establishes that asset path early.

### Milestone 3: Tiny 3D Autoencoder

Goal:

Train a model that compresses a 3D object into a latent vector and reconstructs it.

Build:

- 3D convolution encoder.
- Latent vector, for example 64 or 128 dimensions.
- 3D convolution decoder.
- Binary occupancy reconstruction loss.
- Reconstruction preview after each epoch.
- Reconstruction metric such as IoU.

Study:

- Autoencoders.
- 3D convolution layers.
- Latent vectors as compressed shape descriptions.
- Reconstruction loss and why blurry/soft outputs happen.
- Overfitting a tiny batch as a sanity check.

Resources:

- PyTorch `Conv3d` docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
- PyTorch `BCEWithLogitsLoss` docs: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- PyTorch autograd tutorial: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- Auto-Encoding Variational Bayes paper, useful for next milestone background: https://arxiv.org/abs/1312.6114
- 3D ShapeNets paper, early voxel deep learning context: https://arxiv.org/abs/1406.5670

Artifact:

```text
input shape -> latent vector -> reconstructed shape
```

plus a grid comparing original and reconstructed meshes.

Verification:

- Sanity check: overfit a batch of 8 shapes first; training loss should approach zero and
  those reconstructions should be near-perfect.
- Metric: mean voxel IoU on the validation split above roughly 0.85 at `32^3`.
- Visual check: take a screenshot of the original-vs-reconstruction comparison grid and
  confirm each shape type is still recognizable after the bottleneck.

Devlog angle:

```text
Devlog 3: Teaching a small neural net to remember 3D shapes
```

Why this milestone exists:

This is the first modern core idea. Instead of optimizing a new 3D object from scratch every time, the model learns a compact 3D shape space.

### Milestone 4: Tiny 3D VAE And Latent Interpolation

Goal:

Make the latent space smoother and more sampleable.

Build:

- Mean and log-variance heads.
- Reparameterization trick.
- KL loss.
- Reconstruction plus KL training objective.
- KL warmup so training does not collapse immediately.
- Latent interpolation between two shapes.

Study:

- Difference between autoencoder and VAE.
- Why KL loss regularizes the latent space.
- Posterior collapse at a small scale.
- Why smooth latent spaces matter for generation.

Resources:

- Auto-Encoding Variational Bayes: https://arxiv.org/abs/1312.6114
- Kingma VAE tutorial slides, if you want a lighter explanation: https://dpkingma.com/wordpress/wp-content/uploads/2017/12/nips-2017-vae-tutorial.pdf
- PyTorch VAE example: https://github.com/pytorch/examples/tree/main/vae
- Lilian Weng VAE explanation: https://lilianweng.github.io/posts/2018-08-12-vae/
- Understanding disentangling in beta-VAE, optional: https://arxiv.org/abs/1804.03599

Artifact:

```text
cube latent -> interpolation frames -> cylinder latent
```

or:

```text
sphere -> rounded cube -> box
```

Verification:

- Metric: validation IoU stays close to the plain autoencoder; a small drop is expected.
- Posterior collapse check: the KL term is not near zero across all latent dimensions, and
  reconstructions from sampled (not mean) latents still look correct.
- Visual check: take a screenshot of the interpolation strip — intermediate frames should be
  plausible in-between shapes, not noise or empty grids.

Devlog angle:

```text
Devlog 4: From memorizing shapes to sampling a 3D latent space
```

Why this milestone exists:

A plain autoencoder mostly reconstructs. A VAE starts making the latent space usable for generation, which is closer to the direction of modern 3D systems.

### Milestone 5: Latent Generator With Flow Or Diffusion

Goal:

Generate new 3D latents, then decode them into shapes.

Build:

- Freeze or reuse the trained decoder.
- Train a small latent generator.
- Recommended first version: class-conditioned rectified flow with an MLP.
- Alternative simpler version: class-conditioned latent diffusion with an MLP denoiser.
- Condition only on shape type at first (`sphere`, `cube`, `cylinder`, ...); full attribute
  conditioning (size, descriptor, color) arrives in Milestone 6.
- Decode generated latents into occupancy grids and meshes.

Study:

- Why generating in latent space is cheaper than generating full `32^3` grids directly.
- Diffusion models at a tiny scale.
- Rectified flow as a simpler generative path to experiment with.
- Class conditioning: telling the model which shape family to generate.

Resources:

- Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239
- The Annotated Diffusion Model: https://huggingface.co/blog/annotated-diffusion
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Rectified Flow paper: https://arxiv.org/abs/2209.03003
- Stable Diffusion latent diffusion paper, useful conceptually: https://arxiv.org/abs/2112.10752

Artifact:

```text
noise + "cube" label -> generated cube-like mesh
noise + "cylinder" label -> generated cylinder-like mesh
```

Verification:

- Generate around 16 samples per shape class and decode them.
- Programmatic: most samples should be non-empty, form a single connected component, and have
  filled-voxel counts in a plausible range for their class (compare `dataset_stats.json`).
- Visual check: take a screenshot of a render grid of generated meshes per class — cubes
  should look cube-like, cylinders cylinder-like, and samples within a class should vary.

Devlog angle:

```text
Devlog 5: Generating 3D shapes in latent space instead of optimizing them one by one
```

Why this milestone exists:

This is the toy version of modern native 3D generation. You are no longer only reconstructing known shapes; you are sampling new ones from a learned distribution.

### Milestone 6: Tiny Text Conditioning

Goal:

Make the system respond to simple text without using a huge language model.

Build:

- Small vocabulary parser over the Milestone 1 label vocabulary: colors (`red`, `green`,
  `blue`, `yellow`, `cyan`, `orange`), sizes (`small`, `medium`, `large`), descriptors
  (`tall`, `wide`, `metallic`, `matte`), and shape types.
- Map parsed words to condition vectors, for example concatenated attribute embeddings.
- Conditioning injection into the latent generator.
- Optional: classifier-free guidance by randomly dropping the condition during training and
  blending conditional/unconditional predictions at sampling time. Cheap to add and very
  instructive.
- Invalid prompt handling.
- Prompt metadata saved with generated assets.

Study:

- Difference between labels, tokens, and embeddings.
- Why real models use text encoders.
- Why a toy vocabulary is enough for a low-cost first version.
- How conditioning controls generation.

Resources:

- PyTorch Embedding docs: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- CLIP paper, useful context but not required for MVP: https://arxiv.org/abs/2103.00020
- Classifier-free guidance paper, useful context for conditional generation: https://arxiv.org/abs/2207.12598
- Hugging Face tokenizers course, optional background: https://huggingface.co/learn/nlp-course/chapter6/1

Artifact:

```text
"red sphere" -> red sphere-like mesh
"blue tall cylinder" -> blue tall cylinder-like mesh
```

Verification:

- Programmatic: for prompts with `tall`/`wide`, check the bounding box aspect ratio of the
  generated grid matches the word; check prompt metadata is saved next to the asset.
- Check unknown words fail with a clear error instead of generating garbage.
- Visual check: take a screenshot of a small prompt gallery (one render per prompt) and
  confirm each mesh matches its prompt.

Devlog angle:

```text
Devlog 6: Building tiny text-to-3D without a giant language model
```

Why this milestone exists:

This gives you text-to-3D behavior while staying cheap. It also makes the project easier to explain to non-specialists.

### Milestone 7: Color And Simple Material Channels

Goal:

Move from shape-only generation to asset-like generation.

Build:

- Extend the procedural dataset generator to emit RGB voxel grids. The Milestone 1 dataset
  stores color only as a label; the `COLOR_RGB` palette in `tiny3dlatent/data/labels.py`
  already defines the targets, and a single uniform color per object is enough.
- RGB voxel channels.
- Optional material channels: roughness and metallic (driven by the `metallic`/`matte`
  descriptor labels, which do not affect geometry today).
- Separate shape decoder and material decoder.
- Color/material losses.
- Basic GLB material export where possible.
- Material summary in metadata.

Study:

- Difference between geometry, texture, and material.
- PBR basics: base color, roughness, metallic, normal.
- Why Hunyuan3D and TRELLIS care about texture/material quality.
- Why shape generation and texture generation are often separated.

Resources:

- glTF material documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials
- LearnOpenGL PBR theory: https://learnopengl.com/PBR/Theory
- trimesh material/export docs: https://trimesh.org/trimesh.visual.material.html
- Hunyuan3D-2.1 GitHub, for shape + PBR texture pipeline context: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- TRELLIS.2 project page, for PBR attribute context: https://microsoft.github.io/TRELLIS.2/

Artifact:

```text
"shiny red sphere" -> mesh + material metadata + material render grid
```

Verification:

- Programmatic: load the exported GLB with trimesh and assert a material exists, its base
  color is close to the palette entry in `tiny3dlatent/data/labels.py`, and roughness/metallic
  values match the descriptor.
- Visual check: take a screenshot of the colored render grid — color should match the prompt
  and be uniform across the surface.

Devlog angle:

```text
Devlog 7: Why 3D generation is not finished when the shape looks right
```

Why this milestone exists:

Modern systems target usable assets, not only geometry. Simple material generation makes your project closer to current workflows without requiring a huge texture model.

### Milestone 8: Renderer, Metrics, And Failure Gallery

Goal:

Inspect generated shapes consistently and explain failure modes.

Build:

- Turntable renderer.
- Front/side/top renders.
- IoU metric against procedural targets where labels allow it.
- Connected component count.
- Face count and vertex count.
- Bounding box checks.
- Failure gallery with notes.
- HTML generation report.

Study:

- Why visual quality and mesh quality are different.
- Common voxel generation failures: holes, floating chunks, collapsed shapes, over-smoothed geometry.
- Why consistent rendering matters for comparison.
- What metrics are useful and what they miss.

Resources:

- trimesh repair/inspection docs: https://trimesh.org/trimesh.repair.html
- scikit-image region properties, useful for connected components: https://scikit-image.org/docs/stable/api/skimage.measure.html
- PyTorch metrics can be custom; IoU reference concept: https://en.wikipedia.org/wiki/Jaccard_index
- Blender Python API, optional for higher-quality turntables: https://docs.blender.org/api/current/
- pyrender docs, optional lightweight renderer: https://pyrender.readthedocs.io/en/latest/

Artifact:

```text
generation_report.html
```

with renders, mesh stats, reconstruction metrics, and failure examples.

Verification:

- Programmatic: the report script runs end-to-end from saved outputs without manual steps.
- Visual check: open `generation_report.html` in a browser and take a screenshot — renders,
  metric tables, and the failure gallery should all be present and populated.

Devlog angle:

```text
Devlog 8: Measuring generated 3D assets instead of only eyeballing them
```

Why this milestone exists:

This turns the project into a rigorous engineering artifact. You can show what works, what fails, and how you know.

### Milestone 9: Higher Resolution Or Compact Representation

Goal:

Improve detail without making the project too expensive.

Choose one path:

- Move from `32^3` to `64^3`.
- Add sparse active voxels.
- Add a triplane decoder.
- Add a TSDF decoder instead of occupancy only.

Recommended path:

```text
32^3 dense -> 64^3 dense -> triplane or sparse representation only if needed
```

Study:

- Memory cost of dense voxel grids.
- Why sparse representations matter.
- Why triplanes became popular in neural 3D systems.
- Tradeoff between implementation complexity and visual detail.

Resources:

- EG3D paper, triplane representation context: https://arxiv.org/abs/2112.07945
- Plenoxels paper, sparse voxel context: https://arxiv.org/abs/2112.05131
- Instant-NGP paper, multiresolution grid context: https://arxiv.org/abs/2201.05989
- PyTorch sparse tensor docs, optional: https://pytorch.org/docs/stable/sparse.html
- OpenVDB overview, optional context for sparse volumes: https://www.openvdb.org/documentation/doxygen/overview.html

Artifact:

```text
32^3 output vs 64^3 output comparison
```

or:

```text
dense voxel decoder vs compact decoder comparison
```

Verification:

- Visual check: take a screenshot of the side-by-side `32^3` vs `64^3` (or dense vs compact)
  comparison render — the upgraded output should be visibly sharper.
- Programmatic: record memory use and seconds per training step for both settings in a small
  table; the cost increase should be measured, not guessed.

Devlog angle:

```text
Devlog 9: The cost of detail in 3D generation
```

Why this milestone exists:

This gives you a natural bridge from toy models toward modern representation questions without requiring a massive training run.

### Milestone 10: Final Demo, Architecture Write-Up, And Comparison To SDS

Goal:

Package the project as a clear low-level modern 3D generation demo.

Build:

- Final CLI.
- Small web gallery.
- Generated mesh examples.
- Architecture diagram.
- Failure gallery.
- Cost/runtime table.
- Comparison with the original SDS plan.

Study:

- Optimization-based text-to-3D vs native 3D latent generation.
- What your toy project captures from TRELLIS/Hunyuan-style systems.
- What your toy project intentionally leaves out.
- Which parts would need to scale for real-world quality.

Resources:

- DreamFusion paper, for SDS comparison: https://arxiv.org/abs/2209.14988
- TRELLIS.2 project page: https://microsoft.github.io/TRELLIS.2/
- Hunyuan3D-2.1 GitHub: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- Step1X-3D GitHub: https://github.com/stepfun-ai/Step1X-3D
- TripoSG GitHub: https://github.com/VAST-AI-Research/TripoSG

Final artifact:

```text
text label -> latent generation -> 3D grid -> mesh -> turntable -> report
```

Verification:

- Run the final CLI end-to-end in a clean environment (fresh clone or fresh venv): one command
  from text label to exported mesh and report.
- Visual check: take a screenshot of the web gallery showing several prompts and their
  generated assets.

Devlog angle:

```text
Devlog 10: What I learned by building a tiny modern 3D generator from scratch
```

Why this milestone exists:

The final write-up ties the project to the wider field. It explains why the toy version matters and how it relates to current 3D generation systems.

### Milestone 11: Small Real 3D Dataset Ingestion

Goal:

Learn the practical work required to move from clean procedural shapes to messy real 3D assets.

Build:

- Download or collect a small set of 20-100 public meshes.
- Track source URLs and licenses for every asset.
- Inspect mesh formats such as OBJ, GLB, FBX, and STL.
- Normalize scale so every object fits into the same unit box.
- Center meshes at the origin.
- Repair simple mesh issues where possible.
- Convert meshes into `32^3` or `64^3` occupancy/TSDF grids.
- Create labels such as `cup`, `chair`, `bottle`, or `mug`.
- Compare real mesh voxelization against procedural shape voxelization.
- Fine-tune or test the autoencoder on the small real dataset.

Study:

- Mesh cleanup.
- Scale normalization.
- Coordinate conventions.
- Mesh watertightness.
- Voxelization.
- Dataset licenses.
- Why real 3D datasets are inconsistent.
- Why large 3D datasets increase compute and preprocessing cost.

Resources:

- trimesh loading and processing docs: https://trimesh.org/
- trimesh voxel docs: https://trimesh.org/trimesh.voxel.html
- Blender Python import/export docs: https://docs.blender.org/api/current/bpy.ops.import_scene.html
- Objaverse paper: https://arxiv.org/abs/2212.08051
- Objaverse website: https://objaverse.allenai.org/
- ShapeNet paper: https://arxiv.org/abs/1512.03012
- Creative Commons license overview: https://creativecommons.org/share-your-work/cclicenses/

Artifact:

```text
real_dataset_report.html
```

with:

- source/license table
- before/after normalization renders
- voxelized previews
- mesh issue examples
- procedural vs real dataset comparison
- notes on what broke

Verification:

- Check every real asset has a source URL and license recorded in the report table.
- Programmatic: all normalized meshes fit the unit box and are centered; voxelized grids are
  non-empty.
- Visual check: open `real_dataset_report.html` in a browser and take a screenshot —
  before/after normalization renders and voxel previews should be present.

Devlog angle:

```text
Devlog 11: What broke when I moved from procedural shapes to real 3D data
```

Why this milestone exists:

Procedural data is clean and cheap, but real 3D data teaches the practical problems behind modern 3D generation: broken meshes, inconsistent scale, messy topology, licensing limits, large downloads, and more expensive preprocessing. This milestone belongs after the toy pipeline works so dataset complexity does not block the core model.

## Minimum Viable Demo

The first serious demo should be:

```text
Input: "red sphere"

1. Generate or load procedural shape data.
2. Train a tiny 3D autoencoder on `32^3` occupancy grids.
3. Train a class-conditioned latent generator.
4. Generate a new latent from the label.
5. Decode it into a 3D grid.
6. Extract a mesh with marching cubes.
7. Render a turntable.
```

## Suggested 8-Week Schedule

| Week | Target | Output |
|---|---|---|
| 1 | Skeleton + procedural dataset previews | smoke run + generated shape grid + Devlogs 0-1 |
| 2 | Occupancy/TSDF representation + marching cubes | mesh export from generated grids |
| 3 | Tiny 3D autoencoder | reconstruction results |
| 4 | Tiny 3D VAE | latent interpolation |
| 5 | Latent flow/diffusion generator | new sampled shapes |
| 6 | Toy text conditioning | text label to generated mesh |
| 7 | Color/material channels + metrics | asset-style reports |
| 8 | Gallery + write-up | final demo |

## Optional Weeks 9-10: Real Dataset Extension

| Week | Target | Output |
|---|---|---|
| 9 | Collect and normalize 20-100 real meshes | source/license table + normalized mesh previews |
| 10 | Voxelize real meshes and test/fine-tune model | real dataset report + Devlog 11 |

## Devlog Template

Use this after every milestone.

```md
# Devlog N: Title

## Goal

What I wanted this milestone to prove.

## What I Built

The concrete code, scripts, outputs, and files created.

## What I Studied

The papers, docs, or concepts I used.

## Key Idea In Plain English

The simplest explanation of the technical idea.

## Main Result

Screenshots, renders, meshes, metrics, or videos.

## What Failed Or Confused Me

Bugs, bad outputs, unclear concepts, wrong assumptions.

## What This Teaches About Modern 3D Generation

How this milestone connects to systems like TRELLIS, Hunyuan3D, Step1X-3D, or TripoSG.

## Next Step

What the next milestone builds on top of this.
```

## What To Avoid

- Do not train on Objaverse at the start; use a tiny curated subset only after the procedural pipeline works.
- Do not implement a huge DiT.
- Do not depend on cloud GPUs for the first milestone.
- Do not start with text-to-anything from a large diffusion model.
- Do not try to match TRELLIS/Hunyuan visual quality.
- Do not skip mesh extraction; the asset path is important.

## How This Connects To Your Existing Project

Your current plan:

```text
DreamFusion-style:
text -> 2D diffusion guidance -> optimize voxels -> mesh
```

This new plan:

```text
modern toy native-3D:
synthetic 3D data -> 3D latent model -> generated voxels/TSDF -> mesh
```

Post-MVP extension:

```text
small real mesh set -> cleanup/normalize/voxelize -> test or fine-tune tiny model
```

Both teach useful things, but they teach different eras of the field.

If you want fundamentals of differentiable rendering and SDS, keep the current plan.

If you want a low-level project aligned with current systems, use this new plan.

## Sources Behind The Direction

- TRELLIS.2: O-Voxel, sparse 3D VAE, DiT/flow-style generation, PBR attributes.
- Hunyuan3D-2.1: shape generation plus PBR texture generation.
- Step1X-3D: geometry VAE-DiT plus texture synthesis.
- TripoSG: learned geometry representation and rectified-flow generation.
