# Devlog 7: Why 3D generation is not finished when the shape looks right

## Goal

Move from shape-only generation to asset-like generation: prompts like "red metallic
sphere" should produce a GLB with a correct PBR material, not just a mesh.

## What I Built

- `ColoredOccupancyDataset`: grids gain five painted channels — RGB from the `COLOR_RGB`
  palette plus per-voxel roughness/metallic from the descriptor (`metallic` -> 0.3/1.0,
  `matte` -> 0.9/0.0, geometry descriptors -> neutral 0.6/0.0) — all masked by occupancy.
- `ColorVAE` (`tiny3dlatent/models/color_vae.py`): 6-channel encoder, **separate** shape
  and color decoders sharing nothing but the latent, and a small material head predicting
  `(roughness, metallic)` from the latent. Losses: occupancy BCE + occupancy-masked color
  MSE + material MSE + KL.
- GLB export with PBR materials (`export_glb_with_material` / `load_glb_material` in
  `representation/export.py`) using trimesh's `PBRMaterial`.
- `asset_generate_cli.py`: prompt -> conditioned flow -> ColorVAE `decode_full` -> mesh +
  uniform base color (mean RGB over occupied voxels) + predicted material -> `.glb` with a
  `.material.json` summary next to it, plus a gallery rendered in the *predicted* colors.
- `train_text_flow.py` now trains on either VAE via `build_vae_from_checkpoint`
  (`vae_checkpoint: "latest-color"`), writing `*-color-text-flow` runs.

## What I Studied

- glTF 2.0 PBR materials: baseColorFactor, roughnessFactor, metallicFactor.
- Why Hunyuan3D/TRELLIS separate shape and texture decoding — and how cleanly that maps to
  two decoders over one latent even at toy scale.
- Information flow in autoencoders: a latent can only predict what the encoder saw.

## Key Idea In Plain English

Color and material become extra "paint" channels on the voxel grid. The VAE compresses
geometry + paint into one latent; one decoder rebuilds the shape, another rebuilds the
paint, and a third tiny head reads off the material parameters. The text prompt steers the
flow model to a latent whose paint matches the requested words.

## Main Result

(`outputs/runs/20260611-094638-color-vae/`, `outputs/runs/20260611-104449-color-text-flow/`,
`outputs/runs/20260611-104557-asset-generation/`)

- ColorVAE: geometry val IoU 0.840 (identical to the shape-only VAE — color channels cost
  nothing), color MAE 0.014, material MAE 0.048.
- 24/24 generated GLBs load with a PBR material; 24/24 metallic values correct; 24/24
  roughness on the correct side of the metallic/matte divide; 21/24 base colors within 0.15
  of the palette entry.
- Asset gallery renders in predicted colors: red spheres, green cubes, yellow rounded
  boxes, orange capsules all match their prompts.

## What Failed Or Confused Me

- **The first ColorVAE could not learn material at all** (MAE 0.21 — exactly the
  predict-the-mean floor). The material head reads the latent, but metallic/matte changes
  neither geometry nor color, so the 4-channel encoder input carried zero bits about it.
  Fix: make roughness/metallic *input* channels (6 total). MAE dropped to 0.048. Lesson: a
  latent can only contain what the encoder is shown — obvious in hindsight, invisible until
  the metric called it out.
- Through the generation path, predicted roughness compresses toward the dataset mean
  (matte decodes ~0.7 instead of 0.9) and cyan drifts toward blue on some toruses. The VAE
  recon path is accurate, so this is the flow placing latents slightly off-manifold;
  stronger guidance and 3x flow epochs only marginally helped.
- Prompt "blue metallic tall cylinder" is invalid by construction — `metallic` and `tall`
  are both descriptors and the dataset allows one per object. The parser caught my own
  default prompt.

## What This Teaches About Modern 3D Generation

Shape/texture separation is real architecture, not branding: the two-decoder design mirrors
Hunyuan3D's shape-then-texture pipeline. And the material-channels bug is a miniature of a
real scaling problem — production 3D VAEs must decide which attributes get encoded into the
latent versus conditioned at decode time, and that decision silently caps what the
generator can ever control.

## Next Step

Milestone 8's report (already built) keeps these checks honest; Milestone 9 tests whether
`64^3` fixes the small-feature failures the report surfaced.
