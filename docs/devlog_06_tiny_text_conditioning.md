# Devlog 6: Building tiny text-to-3D without a giant language model

## Goal

Make the generator respond to simple prompts like "blue tall cylinder" using only the
Milestone 1 label vocabulary — no tokenizer, no text encoder, no language model.

## What I Built

- `tiny3dlatent/text/parser.py`: a vocabulary parser over the dataset's label words
  (6 shape types, 6 colors, 3 sizes, 5 descriptors). Handles the multi-word "rounded box",
  rejects unknown words with a clear error listing the known vocabulary, rejects conflicting
  words ("red blue sphere"), and maps parsed attributes to embedding indices with an
  "unspecified" index per category.
- `ConditionedLatentFlow` in `tiny3dlatent/models/flow.py`: the rectified flow now takes four
  attribute embeddings (shape, color, size, descriptor) instead of one class embedding. Each
  table has an extra null row used both for unspecified prompt words and for classifier-free
  guidance.
- `tiny3dlatent/models/train_text_flow.py`: trains on (latent, attributes) pairs from the
  frozen VAE, randomly dropping the whole condition to null 10% of the time (CFG training).
- `tiny3dlatent/models/text_generate_cli.py`: parses prompts, samples with CFG
  (`v = v_uncond + g * (v_cond - v_uncond)`, default g=2.0), decodes, meshes, saves a
  `.prompt.json` metadata file next to every OBJ, renders a prompt gallery, and checks
  bounding-box aspect ratios for `tall`/`wide` prompts.
- `configs/text_flow.json`, `tests/test_text_conditioning.py`.

## What I Studied

- Labels vs tokens vs embeddings — and why a closed vocabulary makes the "text encoder" a
  lookup table.
- Classifier-free guidance: train one model both conditionally and unconditionally, then
  extrapolate between the two predictions at sampling time.
- How conditioning signals reach the generator (concatenation into the velocity MLP input).

## Key Idea In Plain English

The prompt "blue tall cylinder" is just three known words. Each word selects a row in a
small learned table; the rows are concatenated into a condition vector the flow model sees
at every integration step. Guidance turns the condition up louder than it was at training
time: move in the direction the condition changed the prediction, twice as far.

## Main Result

(`outputs/runs/20260610-194302-text-flow/` and `outputs/runs/20260610-194345-text-generation/`)

- All 6 gallery prompts produce 8/8 non-empty shapes that match their prompt visually.
- Aspect-ratio checks: "blue tall cylinder" **8/8** taller than wide; "green wide rounded
  box" **8/8** wider than tall.
- Unknown words fail loudly: `"purple dragon"` raises
  `unknown word 'purple'; known words: blue, capsule, ...`.
- Every exported mesh has a `.prompt.json` with the prompt, parsed attributes, guidance
  scale, steps, and seed.

Also in this round, the VAE was retrained for 80 epochs (vs 40). Honest finding: mean-latent
val IoU did **not** improve (0.860 vs 0.870 — within MPS run-to-run noise; the curve
oscillates around 0.83-0.85 from epoch ~25 on), though sampled-latent IoU rose from 0.814 to
0.833. The decoder is capacity/resolution-bound, which is Milestone 9's problem to solve.
Both flows retrain in ~15s, so swapping the VAE checkpoint is cheap.

## What Failed Or Confused Me

- My "free IoU from more epochs" hypothesis was wrong — worth knowing that the bottleneck is
  architecture/resolution, not training time.
- Size words are subtle in the output: "small" vs "large" shifts the voxel count correctly,
  but at gallery render scale the difference reads less strongly than shape or aspect words.

## What This Teaches About Modern 3D Generation

Conditioning machinery is identical in shape to the big systems: real models swap the lookup
table for a CLIP/T5 encoder and concatenation for cross-attention, but "embed the condition,
inject it into the denoiser/velocity network, amplify with CFG at sampling time" is exactly
what Hunyuan3D and TRELLIS do. The closed vocabulary is the honest toy version of the same
mechanism.

## Next Step

Milestone 7: make the dataset emit RGB voxel grids (colors currently exist only as labels),
add color/material decoding, and export GLB with real materials — turning shapes into assets.
