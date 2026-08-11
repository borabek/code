# Final result — thesis-faithful (2026-07-19)

This is the clean, top-level summary. It follows the Scheffler master thesis exactly and nothing
more. Everything thesis-external (see the last section) is dropped.

## What the thesis does
Segment a terminal-block 3D model into 5 vertex classes with DiffusionNet
(Housing / Contact / SnapPoint / CableEntry / LabelSurface), then localise each feature's
position + normal from its segmented region (§5.3.6, Abb.44). Thesis segmentation score: **0.51
Jaccard**.

## What we did (same method) and the result
- **Data:** Scheffler-Bründl human-labelled corpus (3-rater majority), matched to the exact WSCAD
  STEP. Leakage-safe frozen split: 71 train / 20 val / **11 locked test** (`split_manifest.json`).
- **Model:** DiffusionNet, 3 blocks, width 64, 64 eigen, xyz input, NLL (thesis config).

### Result 1 — segmentation (THE thesis metric), locked-11, single shot
**accuracy 0.858 · mean Dice 0.805 · mean IoU 0.680** → **beats the thesis 0.51.**
This is the one solid, defensible result. (`results/scheffler_semantic/locked_test_result.json`)

### Result 2 — CP extraction (thesis §5.3.6 post-processing)
From the segmented **CableEntry** region → connection point = opening midpoint v_o, direction =
outward axis (manufacturer InsertDirections are 100% axis-aligned, so we snap to the nearest axis).
- Region-derived CP-F1: **val 0.989 @5mm** (0.879 @3mm), position median 0.6mm (`cp_evaluate.py`).
- Manufacturer-validated definition: CableEntry count matches the manufacturer connection count on
  cleanly-labelled parts (0444048 2=2, 0446017 2=2).

## Honest caveats (do not overstate)
- The CP GT is AUTO-derived from human REGION labels, not human-clicked CP points -> call it
  "region-derived CP consistency", NOT a human-CP benchmark.
- The locked-11 was diagnosed during CP work -> for CP it is a dev/regression set, not untouched.
- The segmentation result (Result 1) is the clean, untouched, thesis-comparable number.

## Explored but NOT pursued (thesis-external — dropped by decision 2026-07-19)
- **JSON in-domain CP training** (`train_json_cp.py`, `json_cp_*`): uses the manufacturer JSON
  corpus, which is NOT the thesis data. It failed anyway (dense ~20 CPs/part fragment the region
  labels -> F1 ~0.1). Kept only as an archived experiment.
- **Heatmap + peak detection**: a different keypoint paradigm, NOT in the thesis. Never built.
Both are outside the thesis method; the thesis uses segmentation + region centroid only.

## Bottom line
The thesis is reproduced and beaten on its own metric (segmentation 0.680 IoU vs 0.51), with the
CP extraction layer working on CableEntry. That is the deliverable. The rest was scope creep and
is set aside.
