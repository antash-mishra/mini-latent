# Devlog 8: Measuring generated 3D assets instead of only eyeballing them

## Goal

One command that turns the whole pipeline's state into an inspectable artifact: renders,
metric tables, and — most importantly — an honest failure gallery.

## What I Built

- `tiny3dlatent/report/report_cli.py`: runs the full evaluation end-to-end from the latest
  checkpoints — per-example val reconstruction IoU (not just the mean), fresh
  class-conditional generation with structural checks, fresh prompt generation with aspect
  checks, mesh-quality stats (vertices/faces/watertightness/components/bounds), and a failure
  gallery — then writes `generation_report.html` plus a machine-readable `report.json` into a
  run folder.
- `tiny3dlatent/report/html.py`: small hand-rolled HTML/CSS template; images are copied into
  the run folder so the report is self-contained.
- `save_turntable_strip` in `representation/preview.py`: 8-azimuth render strip per mesh.
- `generate_cli` now records per-sample stats (filled voxels, connected components, in-range
  flag, mesh file) instead of only per-class aggregates, which is what makes a failure
  gallery possible.

## What I Studied

- Why mean metrics hide failure modes: mean val IoU is 0.86, but the per-example
  distribution has a tail down to 0.40.
- Common voxel-generation failure taxonomy: holes, floating chunks, collapsed shapes,
  over-smoothing — and which checks catch which (components catches floaters, voxel-range
  catches collapse, IoU catches smoothing).
- Why consistent camera/lighting matters when comparing renders across runs.

## Key Idea In Plain English

A generative model demo shows you its best outputs; an engineering report shows you its
worst. The report ranks every validation reconstruction by IoU and puts the bottom of the
distribution on the page, with notes, next to the structural-check tables for freshly
generated samples.

## Main Result

(`outputs/runs/20260611-094647-generation-report/`)

- Reconstruction: mean val IoU 0.860 across 200 examples, min 0.40, max 0.98.
- Class generation: 96/96 samples non-empty, single-component, and in voxel range.
- Mesh quality: all six sampled meshes watertight, single-component, bounds inside the unit
  box, 600-4000 faces.
- Prompt checks: 4/4 aspect-correct for both "tall" and "wide" prompts.
- Failure gallery: every worst reconstruction is a *small torus or thin capsule* — at `32^3`
  the hole of a small torus is only 1-2 voxels wide, so the decoder either fills it in or
  breaks the ring. That is the clearest possible motivation for Milestone 9's resolution
  work, found by the report rather than by staring at random samples.

## What Failed Or Confused Me

- No generated sample failed the structural checks this run, which made the "generation
  failures" half of the gallery empty. Good news, but it means the checks are now too easy
  for the model; the report notes this explicitly instead of pretending the gallery proves
  robustness.
- Browser-grade HTML from Python string templates is fine at this scale; the temptation to
  add a templating dependency was not worth it.

## What This Teaches About Modern 3D Generation

Serious 3D systems publish quantitative tables (CD/IoU/FID-style metrics) *and* qualitative
galleries because each catches what the other misses. The per-example IoU tail finding —
"small toruses are the failure mode" — is exactly the kind of insight aggregate metrics
bury, and it changed what I prioritize next.

## Next Step

Milestone 9: measure the cost of moving to `64^3` (the benchmark says 8x per example) and
show whether the extra resolution actually fixes the small-torus failure mode the report
surfaced.
