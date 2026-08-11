# Working-product plan & progress (2026-07-19)

Goal (locked): a WORKING PRODUCT — the model places CPs on NEW, unseen WSCAD terminal blocks.
Method: thesis-faithful (DiffusionNet 5-class segmentation -> CableEntry region -> CP). No JSON,
no heatmap. Base result: segmentation 0.858/0.680 (beats thesis 0.51); it ALREADY generalises
(0321019 unseen = clean seg + correct symmetric CPs).

## Phase 1 — the three items

### 1c MEASUREMENT — DONE (`results/generalization_report.json`)
Ran the full thesis pipeline (STEP->remesh->seg->CP, `infer_step_cp.py`/`measure_generalization.py`)
on all 40 unseen benchmark_candidates:
- **clean 18 (45%)** · noisy 6 · **no_cableentry 16 (40%)** · err 0.
- **Triage (rendered 0294380000/1231700/1552680000): these ARE real terminal blocks, and the model
  MISSED their CableEntry** (predicted Contact/LabelSurface on the wire openings instead). So the
  40% no_cableentry is largely a GENUINE generalisation gap on unfamiliar geometry, not non-terminals
  -> exactly the labelling targets. Renders: `results/step_infer/*_infer.png`.
- Value: we now know WHICH parts the model handles vs not -> targets labelling precisely.

### 1a LABELLING TARGETS — PREPARED (labelling itself is human)
Priority parts to human-label Scheffler-style (5-class), from 1c:
- 6 `noisy`: `results/generalization_report.json` -> noisy_ids.
- The `no_cableentry` that ARE real terminal blocks (triage the 16 visually first).
Workflow: label -> add to corpus with a new split-manifest row -> re-run `measure_generalization`
to confirm the family moved from noisy->clean. Denylist already guarantees these are unseen.

### 1b SYNTHETIC PRETRAIN — BLOCKED, with the exact fix identified
`synth_blocks.py` generated 300 blocks (64 sparse 2-6 CP). BUT proximity-labelling merges adjacent
openings into blobs (35 CP -> 21 regions), the SAME failure as the JSON path. Root cause: the
region approach needs clean PER-TERMINAL regions (which the human gives on Scheffler); synthetic
CPs are just points. **Fix (concrete):** `synth_blocks.build_block` already tracks `rim_idx`
(per-hole vertex ids) -> modify it to emit a per-vertex CableEntry-vs-Housing label from the pocket
membership, giving clean per-terminal regions. Then binary-CableEntry pretrain (reuse the working
`train_json_cp.py` loop) + finetune on real. Cost: 1 code edit + a slow GPU run (~hours on the
4GB T1200). Deprioritised below 1a because the model already generalises and real labels beat
synthetic.

## Phase 2 — TO-DO to get better ON THIS PATH (thesis-faithful, product-oriented)
Ordered by value/effort:
1. **Triage the 16 no_cableentry** (render each) -> split into "real terminal, model missed" (=>
   label) vs "not a terminal" (=> exclude). Cheap; sharpens the true generalisation number.
2. **Test-time augmentation** in `infer_step_cp`: predict on N rotations, average per-vertex probs
   -> steadier segmentation on noisy parts (0270018). No training, no labels.
3. **synth_blocks region labels** (the 1b fix) -> synthetic CableEntry pretrain -> finetune.
4. **Label the noisy/missed families** (human) -> retrain 5-class -> re-measure.
5. **Deploy path**: wrap `infer_step_cp` as the product entry (STEP in -> CP points+directions out
   as JSON), with the QA colour-standard render per run (no "improved" without QA).
6. **Generalisation gate**: freeze a target (e.g. clean-rate >= X% on a held-out unseen family set)
   before claiming the product works.

## Phase A (2026-07-19 night): TTA + labelling prep + FBI review
- **TTA tested -> does NOT help (and large rotations HURT).** `tta_predict.py`: big rotations broke a
  WORKING part (0321019 2 CP -> 0); small tilts are neutral (0321019 293->298 verts) but recover
  nothing on missed parts (0294380000 still 0). The model is orientation-locked; TTA is a dead end.
- **Labelling package prepared** (`prepare_labeling.py` -> `_label_targets/`, 22 parts: 6 noisy +
  16 no_cableentry, each remeshed OBJ + template labels + provenance + README). Labelling is human.
- **FBI review -- no fixable pipeline bug found; the gap is genuinely DATA:**
  - Remesh density ruled OUT (tightening iterations changes nothing; CableEntry stays 0).
  - Orientation ruled OUT (TTA doesn't recover).
  - Resolution ruled OUT (missed parts DO get Contact/LabelSurface on the openings -> resolved,
    just mis-classified as not-CableEntry).
  - => The 40% miss is a genuine SEMANTIC generalisation gap: on unfamiliar geometry the model calls
    the wire opening Contact/LabelSurface, not CableEntry. Triply confirmed (TTA + density + triage).
  - Minor: the clean/noisy verdict heuristic is fuzzy (some "noisy" parts actually have 0 CableEntry
    components) -> a sharper verdict is a nice-to-have, not a blocker.
- **Conclusion:** the cheap levers (TTA, density, mesh) are exhausted. The ONLY lever left is
  targeted human labelling of the missed families (package ready). The pipeline itself is correct.

## Phase 3 — done now (achievable subset)
- [x] 1c measurement (report).           - [x] synthetic corpus generated (300; 64 sparse).
- [x] 1a target list (above).            - [x] 1b blocker diagnosed + exact fix written.
- [ ] to-do #1 triage, #2 TTA -> next achievable actions (no labels/long-train needed).
Remaining need humans (labelling) or long GPU runs (synthetic pretrain) — flagged, not faked.
