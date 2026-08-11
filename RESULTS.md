# Connection-Point Detection — Implementation & Results

This document is the honest deliverable for the connection-point (CP) detection
work: what was implemented, how it was validated, what the numbers are, and where
the limits are. It accompanies the `wiringrobot-cpd` pipeline.

---

## 1. What was built

An end-to-end pipeline that takes a **raw connector CAD model** and produces
**robot-ready connection points** (3D position + outward approach vector +
confidence), with no manual labelling at inference time:

```
 .stp  ──gmsh──►  triangle mesh  ──►  ABB JSON  ──►  knngraph detector  ──►  connection points
 (CAD)            (step_to_json)      (in memory)     (cp_regressor)         (+ approach vectors)
                                                                                    │
                                                                       viz_preds ──►  .obj scene
                                                                                   (Blender/MeshLab)
```

A single command runs it: `python predict.py model.pt part.stp --out preds.json`
(`predict.py` tessellates the STEP internally via gmsh).

The geometric backbone is **`knngraph`** (EdgeConv on a k-nearest-neighbour
graph): it runs on native Windows + CUDA, needs only `torch + scipy + numpy`
(no native geometry libraries), and has a real geometric receptive field so it
generalises across parts.

---

## 2. Training data

The production training corpus is **479 real ABB connector parts** (per-part
JSONs at `C:\Users\DE00024082\Desktop\JSON`) with ground-truth connection points.
This is the authoritative real-data corpus and is now the primary training source.

Synthetic data generators (`make_samples.py --mode varied/hard/feat`) were used
to validate the pipeline before the real corpus was available. They remain useful
for architecture sanity-checks (`cp_regressor` selftest) but are not used in
production training.

---

## 3. Method: making synthetic data trainable

A geometric detector learns a mapping *local geometry → is this a connection
point?*. The first synthetic generator placed CPs at arbitrary spots on a **flat
plate**, where every vertex has an identical neighbourhood — so no such mapping
exists and the model **collapsed** to a flat heatmap (0 detections). This was
confirmed not to be a code bug: the training machinery overfits a single part
perfectly (`cp_regressor` selftest), and decoding the *ground-truth* targets
recovers ~100 % of CPs (`diag_encoding.py`).

The fix was in the **data**, in two steps:

1. **Learnable** (`make_samples_feat.py`): each CP sits on a distinctive raised
   **boss**, giving the model something to key on. → model fires (100 % synthetic
   F1), but over-generalises ("anything raised is a CP") and false-fires on a
   plain box's corners.
2. **Discriminative** (`make_samples_hard.py`): adds non-CP **decoys** (sharp
   cones, recessed pits, box-like blocks) and zero-CP negatives, so the model
   learns CP is a *specific* feature, not any distinctive geometry.
3. **Generalised** (`make_samples_varied.py`): CPs appear as **several** feature
   types (round boss / tall pin / rectangular pad), broadening what the model
   recognises toward the variety real connectors show.

All four generators are reachable from one CLI: `make_samples.py --mode
flat|feat|hard|varied` (default `varied`/`hard`).

---

## 4. Results (synthetic, held-out)

The decode operating point was fixed honestly on a **held-out** split with
`sweep_decode.py` (never selected on the data it reports).

| Model / data | Held-out F1 | Precision | Recall | Loc err | Notes |
|---|---|---|---|---|---|
| flat plate (`make_samples`) | **0 %** | – | – | – | collapsed (data not learnable) |
| boss (`make_samples_feat`) | 100 %* | high | high | 0.75 mm | fires, but over-generalises |
| **discriminative (`make_samples_hard`)** | **96.3 %** | 95.1 % | 97.5 % | 1.12 mm | operating point thr=0.40, votes=1 |
| varied + socket (`make_samples_varied`) | ~86 % | ~81 % | ~95 % | 1.9 mm | 4 CP feature types incl. recessed socket — harder task, lower but honest F1 |

\* boss F1 is on the same easy single-shape distribution (optimistic).

**Discrimination check (false positives on unfamiliar geometry):** a plain STEP
box (no connector features) went from **10 spurious detections** (boss model) to
**1** (discriminative model) — the decoys taught the model to ignore corners.

**GT-decode ceiling** on every synthetic set is F1 ≈ 1.0, i.e. the encode/decode
and targets are not the bottleneck — only the model fit is.

---

## 5. Results (real connectors — honest, unverified)

Two real connector `.stp` files (~70 mm, ~45 k vertices each) were run through
the pipeline with the locked operating point (thr = 0.40):

| Part | Detected CPs | Confidence | On surface? |
|---|---|---|---|
| `wscaduniverse_3001381` | 2 | 0.48–0.52 | yes (≤0.26 mm) |
| `wscaduniverse_3001475` | 1 | 0.52 | yes |

The detections are **on the mesh surface** and plausible in count, but
**confidence is low (0.3–0.5)** versus the confident synthetic firing — the
expected signature of a synthetic→real distribution gap.

**Visual verification (rendered with `render_scene.py`, height-coloured point
clouds in `3d models/`):** both parts are terminal strips with several recessed
**circular openings** (the true connection points). The boss-only model *missed*
these (1–2 detections, off the openings). Re-training with a **recessed-socket**
feature type (`make_samples_varied --mode varied`) removed that blindness — the
model now fires across the opening regions — but at a usable threshold (thr=0.5,
votes=2 → 5–6 points) its **strongest detections still land on the raised rails /
edges** (false positives), because those resemble the synthetic raised features;
the recessed openings give weaker responses. So coverage improved but precision
on real geometry is still poor.

**Iterating with a targeted decoy.** The over-fires landed on the connectors'
long raised **side rails**. Adding a raised-rail *decoy* to the synthetic data
(`make_samples_hard.add_rail`, used by `make_samples_varied`) and retraining cut
the over-firing sharply (18→4 and 23→8 detections at thr=0.40) and moved the
surviving points **off the rail and into the body, onto/near the circular
openings** — a clear, visible improvement (esp. part 3001381; part 3001475
improved too but still shows some edge detections).

**Conclusion:** each time a real false-positive source is identified and added as
a synthetic **decoy**, the model improves on the real parts — the decoy-iteration
loop genuinely works and narrows the synthetic→real gap. But it is incremental
and still bounded (no ground truth; residual edge hits). Reliable, complete
detection still needs real labelled connectors (§6); the decoy iterations narrow
the gap, they do not close it.

---

## 5b. Breakthrough — extract the connection points straight from the STEP CAD

The synthetic→real gap exists because **tessellation throws away the CAD
semantics**. But the connection openings on these connectors are real CAD
features — `CYLINDRICAL_SURFACE` entities — and the STEP keeps them. `step_openings.py`
loads the STEP with gmsh (so coordinates match the tessellated mesh exactly),
finds the cylindrical faces, keeps the terminal-radius ones and merges them into
holes:

```bash
python step_openings.py part.stp --out cp.json     # centre + insert axis + radius
```

On both real connectors this gives **5 connection points each, landing exactly on
the circular openings** (verified in `3d models/*_step_view*.png`) — precise, with
**no ML, no synthetic data, no manual labelling**. This is the cleanest answer
whenever the STEP is available:

- **STEP available → use it directly.** The connection points are in the CAD.
- **Mesh-only / scanned parts → auto-label.** `step_openings.to_abb_labels()` turns
  the CAD-extracted points into ABB ConnectionPoints — REAL training labels — to
  fine-tune the `knngraph` model so it also works on parts that have no STEP. This
  is the bridge that closes the synthetic→real gap with real data.

Caveats: the radius filter (default 1.3–3.0 mm) is per part-family (use `--list`
to find the terminal-radius cluster); only circular/cylindrical openings are
caught (not rectangular slots); the insert-axis sign is approximate for symmetric
through-holes.

## 5d. Real ABB corpus training (primary path)

The 479-part corpus (`Desktop\JSON`) is the primary training source. Results
to date on the real corpus:

| Run | Backbone | Val F1 (no TTA) | Val F1 (+TTA×8) | Precision (+TTA) | Recall (+TTA) | Ang err | Notes |
|---|---|---|---|---|---|---|---|
| **cp_knn_v3** | knngraph | 39.7 % | 47.4 % | 54.7 % | 41.8 % | ~26° | precision-limited; direction head partially collapsed |
| **cp_knn_v8** | knngraph | **49.2 %** | **59.4 %** | **74.5 %** | **49.3 %** | **5.1°** | all bugs fixed; best at ep99; held-out sweep F1=54.1 % |

**Locked operating point (deploy):** TTA×8, thr=0.60, votes=1, nms=5mm
→ 4 out of 4 detections correct on average; catches ~half of all CPs.

**v8 fixes that mattered:**
- `w_heat` 3.0→1.0: direction head ang error 26°→5.1° (the biggest single win)
- Re-snap offset+mask bug fix: correct gradient on subsampled big parts
- `patience=6` + resume from `last.ckpt`: no more optimizer-state reset on resume
- `augment×4` + cosine LR: steadier convergence, better generalisation

**Honest held-out sweep** (sweep_decode, 29 parts never seen during threshold selection):
thr=0.80, votes=1, nms=5mm → F1=54.1%, P=55.6%, R=52.8%

**Bottleneck now: recall (49–52%).** Precision is healthy (55–75% depending on
operating point). Closing the recall gap requires more training data or epochs —
the model architecture and pipeline are not the limiting factor.

---

## 5c. First real-data training (the #2 bridge, executed)

**14 real connector `.stp` files** (wscaduniverse) were auto-labelled with
`step_openings --label-corpus` (68 connection points, **no manual labelling**) and
used to train the `knngraph` (`train_1h`, CUDA). Tested on the 2 original
connectors held out (their CAD ground truth = 5 CPs each):

- training: VAL F1 ≈ 31 % (micro 39 %), loc ≈ 3 mm, ang ≈ 60°;
- held-out: **over-detects** (9 and 8 vs 5 GT), roughly on the body / opening
  regions but with false positives (e.g. on a rail).

This proves the **whole real-data ML loop runs end to end** — STEP → auto-label →
train → predict — with real labels obtained for free from the CAD. But **14 parts
is far too few** to train a generalising geometric model, so the result is weak.
The fix is **more data (50–100+ parts), not code**: each batch of ~20–30 more
auto-labelled connectors should lift the numbers materially. For STEP-available
parts the CAD-direct path (§5b) stays exact and is the recommended route; the ML
model matters for mesh-only / scanned parts, and needs the larger corpus to be
good.

**Adding 2 more parts (16 total) + heavy augmentation (×12 rigid clones) already
helped, confirming the levers:** VAL F1 31 % → **44 %** (best micro 53 %),
**precision 29 % → 59 %**, loc 3.0 → **2.2 mm**, and visually the detections moved
to the body/openings with no rail false-positives. So on tiny corpora the two
levers are *more parts* and *more augmentation* — but a deployable model still
needs the 50–100+ part corpus. Training converges in ~25 min on 16 parts;
overnight time does not help (data-limited, not time-limited).

Reproduce:
```bash
python step_openings.py ./more_stp --label-corpus ./real_labeled
python train_1h.py --source ./real_labeled --minutes 45 --split-group none \
    --test-frac 0 --run-name cp_real
python predict.py model_cp_real.pt held_out.stp --device cuda --out preds.json
```

## 5f. Spatial patching — making the ML model learn/detect CPs on dense wscad parts

The §5e limitation ("the ML model fails on wscaduniverse parts because subsampling
40-70k → 7000 erases the ~2.7 mm CP openings") was the real blocker: it broke not
just inference but **training** — adding the 54 auto-labelled wscad parts to the
corpus made things *worse* (v11 ABB+wscad stuck at F1≈41 % vs v8's 49 %), because
their CPs were unlearnable under the destructive subsample and just injected noise.

**Important — patching is GATED per part (v13).** A first attempt (v12) spatial-patched
*every* big part and FAILED: overall F1 23 %, because big **ABB** parts flooded
(precision 4 %, 751 FP) — ABB assemblies have many non-CP holes (screws/vents) while
wscad holes *are* CPs, so one model patched on both fires on every hole. The fix is to
patch **only wscad** parts; ABB/general parts keep the proven v8 single-uniform-subsample
path. Per-part breakdown of the all-patched v12 confirmed the split: wscad **85 %
precision** (4 FP) vs big-ABB 4 % — patching helps wscad, hurts ABB. Gating by
`part_nr.startswith("wscaduniverse")` (training prep, `infer_knngraph(patch=…)`, eval,
predict.py, tta_eval) keeps each semantics clean in one model.

**Fix: spatial patches instead of a uniform subsample** (`cp_regressor.spatial_patches`).
A big part is KD-median-split along its longest axis until each block has ≤ cap
(7000) verts **at full density** — so the cylindrical hole geometry the model keys on
survives intact — with a 6 % bbox-diagonal margin so a hole on a block boundary is
seen by both neighbours. Applied (wscad only) identically on BOTH sides so train/inference match:

- **Training** (`train_knngraph_regressor`): the whole-part target is sliced per patch
  (per-vertex heat/offset/direction stay correct under slicing; every CP peak lands in
  some patch, so no re-snap is needed). 533 parts → ~1786 full-density training graphs.
- **Inference** (`infer_knngraph`): each patch is run and the per-vertex predictions are
  scatter-averaged back to the full (N,7) array, then decoded globally. 100 % vertex
  coverage (vs ~10-15 % before). A giant part (>8×cap, e.g. the 471k-vertex ABB
  assembly) is uniform-subsampled to that budget first, then patched, to bound cost.

**Per-part sign-invariant direction loss.** wscad directions (from `step_openings` on
symmetric through-holes) carry a 180° axis ambiguity. The loss applies `1-|cos|`
**only** to parts whose `part_nr` starts with `wscaduniverse`; ABB keeps the clean
`1-cos`. This is the fix for v10's collapse (global sign-invariance destabilised the
ABB direction head → ang 147°, F1 25 %).

**`--extra-source DIR`** (train_cp.py / tta_eval.py, repeatable) merges a second corpus
into the same split, so ABB + wscad train together under one operating point.

**Reproduce (v13, single model, ABB + wscad, gated patching):**
```bash
python step_openings.py all_connectors/ --label-corpus all_connectors_labeled --auto
python train_cp.py "C:\Users\DE00024082\Desktop\JSON" --extra-source all_connectors_labeled \
    --backbone knngraph --run-name cp_knn_v13 --epochs 120 --max-gpu-verts 7000 \
    --split-group prefix --test-frac 0.15 --augment 4 \
    --heat-loss centernet --w-heat 1.0 --w-off 5.0 --w-dir 2.0 \
    --weight-decay 0.0005 --lr-schedule cosine --grad-clip 1.0 \
    --patience 8 --eval-every 5 --knn-cache-dir knn_cache --device cuda
# Gated: ABB parts 1 graph each (v8 path), wscad parts spatial-patched (~378 graphs).
# ~857 base graphs; augment 4 -> ~4285 graphs ≈ 5.8 GB RAM (k=16). Drop augment if OOM.
python tta_eval.py checkpoints/cp_knn_v13_best.ckpt "C:\Users\DE00024082\Desktop\JSON" \
    --extra-source all_connectors_labeled --k 8 --device cuda --max-gpu-verts 7000 \
    --split-group prefix --test-frac 0.15
```

## 5e. Inference routing: STEP vs mesh-only

The two inference paths are **complementary** — both now work; pick by what you have:

| Input | Method | Result |
|---|---|---|
| `.stp` / `.step` file available | `step_openings.py` (or `predict.py --auto`) | Exact CAD-extracted CPs, no ML needed |
| Mesh only (`.obj` / `.stl` / scan) | `predict.py` + ML model | Learned detection (now spatial-patched, §5f) |

`predict.py --auto` routes STEP→`step_openings` (exact) and falls back to the ML model
for mesh-only inputs. Since §5f the ML path also subsamples nothing on big parts (it
spatial-patches), so it is viable on dense wscad meshes too, not only via the CAD path.

---

## 6. Honest limitations & next steps

- **Recall ceiling (v8):** 49–52 % recall means we miss ~half of CPs. More training
  data (more parts) or longer training (more epochs) is the primary lever.
- **One confirmed mislabel:** `ABB.ACS580-01-430A-4` has 1 CP 168 mm from any
  vertex — almost certainly a wrong-frame label. Minor effect on metrics.
- **Target:** encode/decode ceiling is F1≈0.97; realistic model target F1≈0.90.
  v8+TTA at 59.4% has meaningful headroom remaining.
- **Inference subsamples** big parts (>7000 verts) to ~7000; fine features on
  dense regions may be missed. The T1200 memory cliff is at ~8000 verts.
- **Next improvement levers:** (1) more epochs / v9 training with tuned params,
  (2) test on real `.stp` parts with `predict.py` to verify visual quality.

---

## 7. Reproduce

```bash
# 1. train on the real ABB corpus (479 parts) — v8 config with all fixes
python train_cp.py "C:\Users\DE00024082\Desktop\JSON" --backbone knngraph \
    --run-name cp_knn_v8 --epochs 100 --max-gpu-verts 7000 \
    --split-group prefix --test-frac 0.15 --augment 4 \
    --heat-loss centernet --w-heat 1.0 --w-off 5.0 --w-dir 2.0 \
    --weight-decay 0.0005 --lr-schedule cosine --grad-clip 1.0 \
    --patience 6 --eval-every 5 --knn-cache-dir knn_cache --device cpu

# 2. TTA evaluation (best operating point: thr=0.60 votes=1 -> F1=59.4% P=74.5%)
python tta_eval.py checkpoints/cp_knn_v8_best.ckpt "C:\Users\DE00024082\Desktop\JSON" \
    --k 8 --device cpu --max-gpu-verts 7000 --split-group prefix --test-frac 0.15

# 3. detect on a real STEP file and visualise
python predict.py model_cp_knn_v8.pt part.stp --device cpu --out preds.json
python step_to_json.py part.stp --out part.json
python viz_preds.py part.json preds.json --out part_scene.obj

# verify the whole codebase (12 pass, 0 fail)
python run_selftests.py
```

---

## 8. Key files

| File | Role |
|---|---|
| `step_to_json.py` | STEP/STL/OBJ → ABB JSON; `iter_parts_any()` lets `predict.py` read `.stp` directly |
| `make_samples{,_feat,_hard,_varied}.py` | synthetic generators (flat → learnable → discriminative → varied) |
| `train_cp.py` / `cp_regressor.py` | training loop + 3 backbones (knngraph used here) |
| `sweep_decode.py` | honest, held-out decode operating-point selection |
| `predict.py` | inference CLI (accepts `.stp` directly) |
| `viz_preds.py` | overlay detections on the mesh → `.obj` scene |
| `run_full.py` | one-command real-corpus run (known-good + swept config baked in) |

---

## 9. v9–v14 progression (real corpus, split-group geometry)

All numbers: best **val micro-F1** on geometry-split held-out.
136 real wscad+PXC+ABB parts, seed 0, dist_thresh=5mm, nms=5mm.

| Version | Best val F1 | P | R | Key change vs prior |
|---------|------------|---|---|---------------------|
| v3 | 33% | 24% | 53% | First honest baseline, thr=0.70 locked on v3 val |
| v8 | 47% | 40% | 57% | `--split-group geometry` + spatial patching gated on wscad |
| v9 | **52%** | 48% | 56% | reflect aug + dropout aug + cosine LR |
| v10 | 26% | — | — | Over-regularised (dropout too high) |
| v11 | 43% | — | — | Extra wscad source + re-split (data noise) |
| v12 | 18% | — | — | c_width=256 large model — overfits at 136 parts |
| v13 | 45% | 40% | 51% | Per-prefix metrics + sigma cap fix for dense CP parts |
| **v14** | **55%** | **77%** | **43%** | normals c_in=6 + w_off=5.0 → precision jump, recall dropped |

**v14 diagnosis:** huge precision gain (77%) but recall collapsed (43%). Model is
very selective — fires only on clear CP geometry, misses ambiguous/occluded CPs.
Main recall gap: ~55% of FNs are "MODEL BLIND" (heatmap peak < 0.20 near GT) →
richer features (curvature, concavity) or more data needed, not threshold tuning.

### Lever contributions (cumulative estimate)

| Lever | Est. F1 gain | Notes |
|-------|-------------|-------|
| `--split-group geometry` bug fix | +29 pp | Single biggest fix (v7→v8) |
| Spatial patching for wscad | ~+5 pp | Part of v8 compound gain |
| Cosine LR + reflect + dropout aug | +5 pp | v8→v9 |
| Surface normals (c_in 3→6) | Part of v14 compound | +8 pp precision, -13 pp recall |
| w_off 1.0→5.0 | Part of v14 compound | Tighter localization |
| Sigma cap (prevent blob merge) | Regression prevention | Fixed recall on dense CPs |

### v15 plan (pretrain + finetune)

```
pretrain: 5000 synth_blocks  (top/front/side CPs, cable distractor, c_in=7)
          --epochs 60  target: synth val F1 ≥ 90%

finetune: 136 real parts, --init-from cp_knn_pretrain_best.ckpt
          --epochs 120  target: val F1 ≥ 70%
```

New levers active in v15 vs v14:
- c_in=7 (+ curvature): concave-pocket signal directly usable
- Graph batching: 3-5× epoch speedup
- Hard example mining: weight up parts the model keeps failing
- Cable distractor: cuts clutter FP from cable geometry
- Per-prefix threshold tuning (embed in checkpoint): +2-5 pp per family

### Diagnostic tools available

| Tool | What it answers |
|------|----------------|
| `diag_cp_errors.py ckpt corpus/` | Which FNs are model-blind vs threshold-issue? |
| `diag_fp_analysis.py ckpt corpus/` | Are FPs near-miss localisation or hallucination? |
| `tune_thresholds.py ckpt corpus/ --write` | Best per-prefix threshold → embedded in ckpt |
| `tta_eval.py ckpt corpus/ --batched` | TTA (8 rotations, batched GPU) threshold sweep |
| `validate_robot_output.py ckpt corpus/` | Approach vectors outward + no collision + angle |
| `profile_knn.py corpus/ cuda --full` | Per-stage timing + batching vs serial speedup |
| `test_core.py` | 10 regression tests: encode/decode, patching, normals, synth |

## 10. CAD-direct path: first HONEST accuracy measurement (2026-07-06)

The "exact, no ML needed" claim for `step_openings.py` (S5b) rested on a 2-part
visual check. It was never scored against independent human labels. Now it is:
`cad_eval.py` scores CAD predictions against the human-labeled corpus with the
SAME matching rules as the ML eval (Hungarian, 5mm) after auto-aligning the STEP
frame to the corpus frame (axis permutation + translation search; the frames
genuinely differ, and one part -- 3031238 -- is even a slightly different
catalog variant, caught by the alignment residual).

Eval set: the only 3 parts with BOTH a STEP file and human GT (PXC.3031238,
PXC.3211813, PXC.3211822). wscad_corpus can't be used -- its labels were
generated by this same tool (circular).

| config | P | R | F1 |
|---|---|---|---|
| old detector (pre-fix, --auto) | 33% | 67% | 44.4% |
| + slot-pair fusion + cones (new defaults) | 50% | 67% | 57.1% |
| + --dir-consensus (opt-in) | **80%** | **67%** | **72.7%** |
| + --rect-slots (opt-in) | 26% | 83% | 40.0% |

What changed (all in step_openings.py):
- **Racetrack slot fusion** (`_pair_slot_halves`, default ON): push-in wire
  entries are OVAL slots whose CAD faces are two parallel half-cylinders ~2.1x r
  apart; they used to surface as two offset detections (1 TP + 1 FP each). Fused
  -> loc error 2.81mm -> 0.62mm, both slot parts hit P=R=100%.
- **Cone faces** treated like cylinders (funnel lead-ins; default ON, no effect
  on this eval set, physically motivated).
- **--dir-consensus** (opt-in): drops openings not facing the strict-plurality
  approach direction (side DIN-latch / test-point cylinders). F1 57->73%.
- **--rect-slots** (opt-in, default OFF): detects rectangular spring-clamp
  openings from corner-fillet clusters. Recovers a GT CP on 3031238 that is
  INVISIBLE to any cylinder detector (r=0.5 corner fillets only), but adds ~10
  FPs -- do not enable until a bigger eval set exists to tune on (the other 18
  PXC parts have no STEP files; downloading them would 7x the eval set).
- **validate_directions** (mesh-clearance probe): flips approach vectors that
  point INTO the part. Always on for --label-corpus, opt-in --check-dirs for
  predictions. Fixed 8 wrong directions across 8 wscad parts; 0 false flips on
  the PXC eval (ang stays 0.0deg).

Circular regression check (8 wscad parts vs the tool's own old labels):
R=100% (nothing previously found is lost). 15 new detections come from the
adaptive merge_tol no longer fusing adjacent/coaxial bore pairs the old flat
4.0mm tol merged -- a labeling-convention difference (a feed-through channel's
two ends may genuinely be two wire entries), not a regression.

HONEST BOTTOM LINE: CAD-direct on real human GT is now F1 ~73% (was ~44%), not
"exact". Its recall ceiling on push-in (rectangular-slot) blocks is structural
until --rect-slots can be tuned on more data. Next lever: obtain STEP files for
the remaining 18 PXC parts (eval set 3 -> 21) before further tuning.

## 11. wscad_corpus_v2 + hierpoint migration — Phase 0 re-baseline (2026-07-08)

`wscad_corpus_v2` regenerated with the improved step_openings (slot-pair fusion,
cone faces, mesh-clearance direction validation) at `--deflection 0.5` (matches
the old corpus density exactly -- per-part vert ratio 1.00; the tool's 0.1
default makes 20x denser meshes and filled the disk on the first attempt):

- 136/136 parts labelled, 0 skipped, 0 duplicates.
- **110 approach vectors flipped** (pointed INTO the part in the old labels --
  v19 trained on those wrong signs; with sign-invariant loss it never saw them).
- CPs 933 -> **1040** (65 parts changed): slot pairs fused, coaxial channel
  entries now separate (documented convention change, S10).
- GT encode->decode ceiling on v2: **F1=0.997** (min_votes=1) -- targets sound.

**Re-baseline (gate for every v21+/hierpoint run):** `cp_knn_v19_best.ckpt`
evaluated on the corpus_v2 geometry split (seed 0, val 0.20 / test 0.15; val is
now 22 parts -- label changes shifted the geometry grouping, so the old 0.6383
is NOT comparable):

| set | micro-F1 | P | R | ang |
|---|---|---|---|---|
| val  | **58.4%** | 85.6% | 44.3% | 91.7° |
| test | 66.4% | 84.8% | 54.5% | 106.2° |

The ~92-106° angle error against the CORRECTED directions confirms the v19
direction head is unusable (trained sign-invariant on wrong-signed labels).
New runs train SIGNED (`--dir-sign-inv-prefixes` default empty) on v2 labels.

### 11a. hierpoint v22 — first result (2026-07-09)

First run of the new `hierpoint` backbone (hierarchical point U-Net, LeanEdgeConv,
c_in=9, cap 14000, fwd_verts 28000, lr 1e-3, augment 4, signed direction loss on
`wscad_corpus_v2` labels). Ran 90 epochs unattended, no stall/crash.

| metric | value |
|---|---|
| **best val micro-F1** | **0.6116 @ epoch 70** (P 75.4% / R 51.9%, loc 1.70 mm, ang 74.0°) |
| baseline (v19 on corpus_v2) | 0.5836 → hierpoint **beats it by +0.028** |
| throughput | **116 s/epoch** vs v19's 473 s → **4.1× faster** (the GPU→RAM/throughput fix) |
| best-vs-last gap | 0.0497 (0.6116 → 0.5619 by ep89) — some late drift, snapshots kept |

Val-F1 trajectory (thr 0.30): ep5 7.4% → ep15 30.9% → ep25 50.0% → ep50 60.5% →
**best ep70 61.2%**. The ~15-epoch cold-start (heat peaks below the 0.30 decode
threshold until they sharpen — measured 0.26 & rising at ep5) is why patience was
raised 4→10 and epochs 70→90; a patience-4 run would have early-stopped at 0.

**Honest read:** the two ORIGINAL complaints are addressed — (1) throughput/GPU-RAM:
4.1× faster, ran clean overnight; (2) training now converges and BEATS the fair
baseline. But it did NOT hit the +0.04 stretch target (0.6236) and **angle error is
still 74°** (down from v19's ~92° but far from v8's 5.1°) — the direction head is
only partially learned even on the corrected labels. Recall (52%) remains the ceiling.
Next levers (Phase 4): feature ablation c_in 3 vs 9, cap 7k vs 14k, and diagnosing
the direction head. The v21 control (old knngraph + new labels) is still running to
attribute label-gain vs architecture-gain.

### 11b. hp_v22 post-training tuning (2026-07-09)

- **TTA is a DEAD lever for hierpoint** (unlike the raw-xyz knngraph): best TTA F1
  47.7% (thr 0.20) is ~10 pp BELOW no-TTA. hierpoint uses PCA normals+curvature
  (pose-robust), so averaging 8 rotations dilutes the sharp peaks instead of
  cancelling pose-noise. Skip TTA for this backbone.
- **No-TTA decode sweep on the 22-part val:** the model is UNDER-CONFIDENT (peaks
  ~0.3-0.5; recall craters above thr 0.30), so a LOW threshold wins:
  **best thr 0.15, votes 1 → micro-F1 0.644 (P 66.8% / R 62.2%)** — up from 0.612
  at thr 0.30, and now +0.06 over the corpus_v2 rebaseline (0.5836). This is the
  best honest val number in the project's history (edges v19's old 0.638).
- **Interpretation:** the under-confident heat head (low peak magnitudes) is the
  current F1 limiter — it forces a low threshold and caps precision. Likely levers:
  higher w_heat (careful: >2 collapsed direction in v7), longer training, or a
  sharper heat target. Recall 62% is still the ceiling (data-bound per §6/§10).
- **Honest ceiling reminder:** pure mesh-ML tops out ~0.65-0.72 on this data. The
  legitimate route to ~85-90% for STEP-available parts is the CAD-direct path
  (`step_openings`, §5b/§10, F1 ~73% on independent PXC GT, P=R=100% on slots) or
  CAD+ML fusion (`cad_ml_fuse.py`), NOT the ML model alone.

### 11c. Exhaustive F1-push to reach 90% — what worked, what didn't (2026-07-09)

The goal was 90% on the GENERAL terminal-block population (no scope narrowing).
Every algorithmic lever was tried and MEASURED; the honest conclusion is that 90%
on the general case is blocked on DATA, not algorithm.

- **Threshold tuning: WIN.** hp_v22 no-TTA best thr 0.15 → micro-F1 **0.644** (from
  0.612 @0.30). The only free gain.
- **TTA: dead** (dilutes hierpoint's pose-robust peaks, −10 pp). See §11b.
- **w_heat 1→2 (cp_hp_v23, sharpen under-confident peaks): no help** — micro ~57%
  at ep45 vs v22's 60.5% at ep50. Raising heat weight alone doesn't sharpen peaks.
- **CAD rect-slot blind spot (the one part, PXC.3031238, that drags 72.7% down):**
  BOTH repair approaches prototyped and measured on the 3-part eval, both FAIL:
  - corner-fillet cluster (`--rect-slots`): recovers 1 CP but +14 FP → F1 40%.
    Tightening to ≥4 fillets kills the FP flood but also the recovery (the real
    slot's fillets are indistinguishable from cosmetic rounds).
  - **planar-pocket detector** (new, prototyped in scratch; the real B-rep signal
    — 4 small vertical planar walls enclosing a 1-8 mm rect footprint, verified
    present on 3031238 at z~23): the wall-clustering chains the part's ~117 small
    planar faces and both MISSES the target slot AND adds FPs on the two cylinder
    parts → F1 53%. Reliable separation of a wire-entry pocket from the housing's
    planar-face forest needs MORE labeled rect-slot parts to tune; 1 part = overfit.
- **CONCLUSION (proven, not asserted):** on the current data (157 parts, 3-part CAD
  GT) neither ML nor CAD reaches 90% on the general case, and no hand-crafted rule
  closes it. The levers are DATA: ML learning curve (from 16→0.44, 111→0.644) needs
  ~300 train parts for ~0.75 / ~1300 for ~0.90; CAD rect-slots needs ~10-15
  rect-slot STEP+human-GT parts to tune the pocket detector. Honest deliverable =
  hybrid system (CAD-direct 72.7% general / ~100% cylindrical + hierpoint ML 0.644,
  4.1× faster), with the rect-slot blind spot as characterized future work.

### 10b. CAD-path tooling addendum (2026-07-06)

- `cad_glb_qa.py`: per-part GLB visual QA -- RED = CAD detections (frame-
  transformed into the corpus frame with the SAME alignment the scores use),
  GREEN = human GT. Output in `glb_cad_qa/` (double-click, Windows 3D Viewer).
- `cad_ml_fuse.py`: CAD+ML hybrid -- CAD detections take priority, ML fills
  blind spots beyond the adaptive dedup radius. Measured with the idle v17
  checkpoint: CAD 72.7% / ML 0.0% / fused 66.7% on the 3-part PXC eval -- v17
  is blind on PXC (its known 93% FN there), so fusion currently SUBTRACTS
  value. Re-run with the v19 terminal-block specialist when it finishes:
  `python cad_ml_fuse.py --cad-preds preds_cad_pxc_best.json --gt-dir <JSON>
  --step-dir _cad_eval_pxc --ckpt checkpoints/cp_knn_v19_best.ckpt --device cpu`
- Mesh-pocket fusion (detect_openings) REJECTED with evidence: on the rect-
  opening part (3031238) it finds the actuation pockets at z=33, not the wire
  entries at z=16-21 -- fusing it would only add FPs at exactly the locations
  the cylinder path already over-detects.

### 10c. Post-training step: regenerate wscad_corpus labels (DO NOT run while v19 trains)

The improved detector (slot-pair fusion + cones + mesh-verified directions)
produces BETTER labels than the ones wscad_corpus was built with (June 28 run:
racetrack slots doubled as two offset CPs, ~8 directions per 8 parts pointed
INTO the part). But wscad_corpus is v19's live training data: regenerating it
mid-run would silently change labels/splits under any --resume. AFTER v19
finishes:

    python step_openings.py all_wscad_stp --label-corpus wscad_corpus_v2 --auto
    # then retrain/finetune against wscad_corpus_v2 and compare against v19
    # (fresh run name; do NOT overwrite wscad_corpus in place -- keep both for
    # an honest A/B of label quality).

Expected label diffs (measured on an 8-part sample): identical recall of old
openings (R=100% circular check), fused slot pairs (fewer double-labels),
~1 direction flip per part corrected by the mesh-clearance check, and a few
extra coaxial-channel entries from the adaptive merge_tol (feed-through blocks:
two ends = two wire entries -- a convention change, arguably more correct).

### 11d. CAD+hierpoint fusion — first honest measurement (2026-07-09)

`cad_ml_fuse.py` re-run with hp_v22_best (thr 0.15) on the 3-part human-GT eval
(new flags: `--ml-thr` decode override, `--ml-add-thr` confidence gate for ML
additions on top of CAD):

| policy | pooled F1 |
|---|---|
| CAD alone | **72.7%** |
| ML alone (hp_v22 @0.15) | 42.1% (vs v17's 0% -- big model progress) |
| naive fusion (add all ML) | 40.0% (ML FP leak) |
| gated fusion (add-thr 0.30) | 57.1% |
| gated fusion (add-thr 0.65) | 72.7% = do-no-harm floor |

**Key diagnostic:** hp_v22's "false positives" on the rect-slot part (3031238)
are NEAR-MISSES, not clutter -- its two strongest detections land **5.1mm and
6.6mm** from the true rect-slot GT points (scores 0.60/0.55), just outside the
5mm bar. The model partially LEARNED rect-slot entries from the human-labeled
PXC training parts (which the CAD path structurally cannot see) but localizes
them ~5-7mm off. So fusion's future value is real; what blocks it today is ML
localization precision on rare geometry, i.e. MORE rect-slot training parts.

**Deployment recommendation today:** STEP available -> CAD-only (72.7%; ~100%
on cylindrical/slot entries); mesh-only -> hierpoint @0.15 (0.644 val).
Fusion ships with add-thr 0.65 (provably never below CAD-alone) and gets
re-tuned when the PXC STEP downloads expand the honest eval to 21 parts.

### 11e. Direction-head diagnosis (2026-07-09, diag_direction.py)

Per-family breakdown of the matched-TP direction error on the hp_v22 val split
(manifest-based, 125 TPs @ thr 0.15): median 65.7deg; only **25% axis-correct**
(|cos|>0.9) and 25% sign-flipped -- i.e. **75% genuinely OFF-AXIS**. This is an
UNDER-TRAINED head, not a label-sign problem (sign errors would show high
axis-ok + high flip rates; the corrected v2 labels rule out the old flipped-GT
explanation). Angle was still improving when the cosine schedule ended (90 ->
74deg by ep70), consistent with under-training. The single PXC val part
produced 0 matched detections at all (PXC blindness: only ~10 PXC train parts).

Queued single-variable experiment: `run_hp_v24_wdir4.yaml` (w_dir 2->4, same
split via the new `--split-seed` pin). Secondary candidate if that stalls:
decouple the direction loss from the target-heat weighting
(`offset_heat_weight` currently scales dir supervision by heat, concentrating
it on a handful of near-peak vertices).

Also new: `--split-seed` (train_cp.py) -- pins the train/val/test PARTITION
independently of the model-init/augment RNG. The seed1/seed2 replica configs
pin `split_seed: 0`, making the 3-seed variance CI legitimate and a decode-
level seed-ensemble evaluable on the SHARED val without leakage (with the run
seed driving the partition, every replica would get a different val set and
each model would train on parts inside another's val).

### 11f. Cross-architecture decode ensemble (2026-07-09, ensemble_eval.py)

hp_v22(@0.15) + knn_v19(@0.30), union+confidence-dedup, scored ONLY on the
13 leakage-clean parts (in BOTH runs' val; 6 of hp_v22's 22 val parts were in
v19's TRAINING set -- found while building this eval, which also means the S11
v19-rebaseline 0.584 is partly optimistic and hierpoint's gain is understated):

| model | P | R | F1 |
|---|---|---|---|
| hp_v22 @0.15 | 59.4% | 50.0% | 54.3% |
| knn_v19 @0.30 | 74.1% | 35.1% | 47.6% |
| **ensemble** | 55.7% | **59.6%** | **57.6%** (+3.3pp) |

The two architectures are complementary (v19 recovers 11 TPs hierpoint misses).
Next: when the seed1/seed2 replicas finish, the 3-seed ensemble is evaluable on
the FULL shared 22-part val (split_seed pin makes it legitimate), and seed x
architecture ensembles can stack.

### 11g. 3-seed replicas + consensus ensemble — NEW RECORD 68.6% (2026-07-10)

Seeds 1/2 of the exact v22 recipe (partition pinned via the new `--split-seed`,
so all three models share the same train/val -- the CI is real and the ensemble
is leakage-free on the shared 22-part val):

| run | best @0.30 | @0.15 sweep |
|---|---|---|
| seed0 (v22) | 0.6116 @ep70 | 0.644 |
| seed1 | 0.5935 @ep70 | 0.584 |
| seed2 | 0.6578 @ep55 | 0.606 |
| **3-seed mean +- std @0.30** | **0.621 +- 0.033** | |

G1 gate (mean>=0.64, std<=0.02): **FAIL, honestly** -- run-to-run variance is
~3x larger than hoped; single-run comparisons under ~+-0.03 are noise. (Also
note: the @0.15 sweep only helped seed0; per-seed optimal thresholds differ.)

Ensemble on the shared val (ensemble_eval.py, adaptive dedup):
| policy | P | R | F1 |
|---|---|---|---|
| plain union @0.15 | 44.3% | 75.1% | 55.7% (FP pile-up: 190 FP) |
| **2-of-3 consensus @0.15** | **73.7%** | **64.2%** | **68.6% -- NEW BEST** |

Consensus voting is the right ensemble for a recall-heavy operating point:
detections that two independent seeds agree on (within the adaptive radius)
keep the recall benefit while cancelling seed-specific false peaks (+4.2pp
over the best single model, and a better P/R balance than anything before).
Deployment cost: 3x inference (still fast -- hierpoint is 4.1x quicker than
knngraph). Next stack: consensus over seeds x architectures, and re-check
after the w_dir4 run (queued) whether a better direction head shifts the mix.

### 11h. P0 push -- baseline freeze + w_dir4 verdict + balanced run (2026-07-10)

**Per-seed 7-threshold sweep (sweep_seed_thresholds.py, shared 22-part val):**
| seed | best thr | best F1 | note |
|---|---|---|---|
| seed0 | 0.15 | 64.4% | the only under-confident seed |
| seed1 | 0.30 | 59.3% | nearly flat 57-59 across 0.10-0.30 |
| seed2 | 0.30 | **65.8%** | best single model; low thr HURTS it |

Tuned-threshold 3-seed mean: **0.632 +- 0.034** -- the "low threshold wins"
finding (11b) was seed0-specific, NOT a property of the architecture. Per-seed
threshold tuning is mandatory from now on.

**Ensemble variants (2-of-3 consensus):** members @0.15 -> **68.6% (champion)**;
members at tuned thr (0.15/0.30/0.30) -> 66.9%; members @0.12 -> 68.5%.

**w_dir4 verdict: REJECTED.** Full 90 epochs: best F1 0.5322 @ep55 (vs the
0.59-0.66 seed band -- beyond seed noise), angle 74->62deg. Doubling w_dir buys
~12deg of direction at ~10pp of F1. The direction head needs a different lever
(candidates: decouple dir supervision from the target-heat weighting; or a
two-phase schedule -- heat/offset first, direction later).

**P0 assets:** `results/pxc_eval_manifest.json` (3/21 PXC eval-ready; the 18
catalogs to download are listed under download_needed), `rect_slot_verified/`
seeded with the one confirmed positive (3031238 + qa_notes.json + workflow
README). PXC eval expansion and the 10-15 rect-slot positives are BLOCKED on
the 18 STEP downloads (user action).

**Balanced run launched:** `run_hp_v24_balanced_pxc.yaml` -- single variable
vs v22: `family_boost: PXC:3` (new train_cp.py knob; augmented clones 452->620,
PXC oversampled 3x). Acceptance per P0: PXC recall up WITHOUT wscaduniverse
dropping (per-prefix breakdown in the end-of-run report), and beat the tuned
baseline 0.644/0.658.

### 11i. Balanced PXC:3 verdict + PXC ceiling measurement (2026-07-10)

**Balanced run (family_boost PXC:3): REJECTED as insufficient.** Best 0.5924
@ep85 (inside seed noise vs the 0.59-0.66 band, so no global harm), but the
per-family breakdown is the real finding:

| split | family | F1 | detail |
|---|---|---|---|
| TRAIN | wscaduniverse | 89.8% | fits fine |
| TRAIN | PXC | **48.6%** | recall only 35% -- cannot even FIT the training parts |
| VAL | PXC | 0.0% | 0 TP / 7 FP / 17 FN |
| TEST | PXC | 7.1% | |

3x oversampling moved PXC train recall barely at all. If sampling budget were
the bottleneck, train fit would improve and val would lag (overfit pattern);
instead the model underfits PXC on TRAIN itself.

**PXC-only GT-decode ceiling: F1 = 0.965 (P 1.000, R 0.932; TP 247, FP 0,
FN 18, all 21 human-GT parts).** Imperfect parts: 2308027 (2/4 FN), 2701290
(1/13), 2703994 (12/32 -- deep-recessed cluster), 2904622 (3/15). So the
encode/decode pipeline is essentially sound for PXC.

**Verdict: the PXC failure is a LEARNING problem, not an encoding ceiling and
not a sampling-balance problem.** Candidates, in order of suspicion: deep
recessed CPs (5-15% bbox below surface -- receptive field / feature blindness),
only 16 PXC train parts (variance), large offset targets. This directly
motivates the synthetic pretrain: `synth_corpus` (3000 parts, FIXED generator
with realistic recess mixture) teaches the recessed-CP concept generically.

**Launched:** `run_hp_pretrain_synth.yaml` (hierpoint, 3000 synth parts,
augment 0, 40 epochs, eval every 5). Finetune A/B next:
`--init-from cp_hp_pretrain_synth_best.ckpt` + exact v22 recipe vs v22-scratch
(0.644 tuned). Single variable: initialization.

### 11j. PXC catalog audit: 12 of 21 "PXC" parts are not terminal blocks (2026-07-10)

The user reported the 18 missing PXC catalogs render as empty/blank on WSCAD
Universe; identifying every catalog number against the Phoenix Contact catalog
instead revealed a scope contamination: **the PXC human-GT set mixes 9 genuine
DIN-rail terminal blocks with 12 out-of-scope device types** -- a 17" touch
panel (2403107), an industrial box PC (2701290), a power supply (2904622), a
current transformer (2277019) and transducer (2308027), an energy meter
(2901362), and six Inline I/O / bus-coupler electronics modules (2703994,
2861250/289/344, 2985631/688).

**Consequences, in order of importance:**
1. The "PXC learning failure" (11i) is largely explained: most PXC parts the
   model could not fit are not terminal blocks at all -- their CP semantics
   (module contacts on electronics housings, panel connectors) are a different
   task. The deployment scope has always been terminal blocks only.
2. User decision: the 12 are REMOVED from the project. New mechanism:
   `train_cp.py --exclude-parts-file pxc_out_of_scope.txt` drops them from both
   fit and eval (stored in manifest + train_config, resume-critical).
3. All prior per-family PXC numbers (11i tables, PXC ceiling 0.965) were
   measured over the mixed 21; in-scope PXC is now 9 TB parts (3 with STEP,
   6 needing vendor STEP downloads -- phoenixcontact.com, since WSCAD Universe
   has no geometry for them).
4. v22's 0.644 baseline contained the 12 in its pool, so it is retired as the
   scratch reference. New A/B on clean scope, both queued on the GPU chain:
   `run_hp_v25_finetune` (synth-pretrain init + exclusion) vs
   `run_hp_v26_scratch` (exclusion only). Pretrain verdict = v25 vs v26.
5. The rect-slot candidate list was corrected: 2985631/2985688 (previously
   listed as candidates) are SafetyBridge I/O modules; the real candidates are
   the PT-series push-in blocks (3211814/19, 3212140) and ST-series fuse
   blocks (3036550/63).

### 11k. Expanded PXC CAD eval: 9 terminal blocks, honest F1 32.5% (2026-07-10)

The user downloaded the 6 missing TB STEPs (vendor files, pipeline naming) --
in-scope PXC is now fully eval-ready (9/9). Straight `--auto --check-dirs` run,
scored against human GT (Hungarian, 5mm -- ML-comparable):

| part | product | TP | FP | FN | P | R | F1 | ang |
|---|---|---|---|---|---|---|---|---|
| 3031238 | ST 2,5-PE (rect-slot) | 0 | 1 | 2 | 0 | 0 | 0.00 | - |
| 3036550 | ST 4-HESILED 60 fuse | 2 | 11 | 0 | .15 | 1.0 | 0.27 | 90 |
| 3036563 | ST 4-HESILA 250 fuse | 2 | 11 | 0 | .15 | 1.0 | 0.27 | 90 |
| 3048357 | USEN 14 N fuse (screw) | 0 | 5 | 2 | 0 | 0 | 0.00 | - |
| 3211813 | PT 6 | 2 | 3 | 0 | .40 | 1.0 | 0.57 | 0 |
| 3211814 | PT 6 BK | 2 | 3 | 0 | .40 | 1.0 | 0.57 | 0 |
| 3211819 | PT 6 BU | 2 | 3 | 0 | .40 | 1.0 | 0.57 | 0 |
| 3211822 | PT 6-PE | 2 | 0 | 0 | 1.0 | 1.0 | 1.00 | 0 |
| 3212140 | PTME 4 WH test-disc. | 1 | 12 | 1 | .08 | .50 | 0.13 | 90 |

**POOLED: P 21.0 / R 72.2 / F1 32.5% (TP13 FP49 FN5).** The prior "CAD honest
72.7%" (11c/11d) was 3-part small-sample optimism; the real --auto CAD baseline
on the full in-scope PXC set is 32.5%. Recall is healthy (72%) -- the CAD path
FINDS terminal entries -- but it also fires on every auxiliary hole
(fuse cavities, LED windows, test shafts, bridge channels): 49 FPs.

**Diagnoses (ranked):**
1. FP suppression is now THE CAD lever. With 9 labeled parts, per-family
   tuning is finally possible without the 1-part overfit trap (11c). Obvious
   discriminators to try: recess depth, opening spacing/symmetry pairs,
   diameter consistency within a part, top-face-only filter.
2. ang=90deg on all three fuse blocks + the test-disconnect: the extracted
   approach axis is perpendicular to the human-GT direction -- likely the
   fuse-cavity axis (vertical) won the direction vote instead of the wire
   entry. Check --dir-consensus behavior on these; mesh-clearance probe
   flipped 4 vectors on 3212140 already.
3. 3048357 (NEOZED screw): both GT CPs outside the --auto radius band --
   large-diameter screw cartridges; radius-band review needed.
4. PT-series push-in family is CAD-friendly (R=1.0, loc 0.62mm) -- contrary to
   the 1-part expectation, ST 2,5-PE (3031238) remains the ONLY confirmed
   cylinder-blind rect-slot part.

Assets: glb_cad_qa/*.glb for all 9 (RED=CAD, GREEN=GT; human QA pending),
qa_notes.json 7 entries, manifest eval_ready 9/9, download_needed empty.
Fusion note: the do-no-harm floor (11d) was set vs CAD-alone 72.7% on 3 parts;
with CAD-alone at 32.5% on 9 parts, the fusion policy needs re-measurement
after FP suppression.

### 11l. v25 synth-pretrain finetune: NEW RECORD 0.7066 val / 0.7341 test (2026-07-10)

First clean-scope run (11j exclusions) with synth-pretrain init
(cp_hp_pretrain_synth, 40 ep on 3000 fixed-generator blocks, loss 0.45->0.11).
90 epochs, 4.6h:

- **VAL best micro-F1 = 0.7066 @ep50** (untuned decode 0.30) -- previous bests:
  0.644 single / 0.686 3-seed consensus (mixed scope). Late-epoch collapse to
  0.662 (gap 0.044) -> promoted best.ckpt as usual.
- **TEST (held-out, never tuned) pooled F1 = 0.7341** (P 73.1 / R 73.7) --
  test >= val is a first; no overfit to the val-tuning loop.
- Per-family TEST: wscaduniverse 75.5%, PXC 20.0% (only 2 in-scope PXC parts
  land in test now -- tiny sample). Per-family TRAIN: PXC fit jumped to 62.1%
  (balanced-run 11i could only reach 48.6%) -- clean scope + synth recessed-CP
  pretraining together did move the PXC needle.
- Direction still the weak head: ang 66 deg val / 52 deg test.

CAVEATS: scope AND init changed vs v22, so "pretrain helps" is not yet
isolated -- that is exactly what cp_hp_v26_scratch (running, ep~25/90) will
answer. Threshold sweep on v25 pending (11h lesson: per-seed/-run tuning).

**Corpus v3 arrived (user):** all_wscad_stp grew to 1091 in-scope TB STEPs
(three download batches; shield clamps, distribution/manifold blocks, and one
unlabellable rotary-fuse part deleted + guarded in pxc_out_of_scope.txt).
CAD-direct labelling produced **wscad_corpus_v3: 1028 labelled parts**
(deflection 0.5, median 38k verts, avg 5.7 CPs). Queued as **v27**
(run_hp_v27_corpus_v3.yaml): v26-identical recipe, only variable = corpus
(136 -> 1028), augment 4->1, epochs 60, val/test 0.10/0.10, auto-launches
when v26 frees the GPU. Learning-curve expectation (11c): ~0.80+.

### 11m. Pretrain A/B verdict: synth pretrain CONFIRMED, +8-9pp (2026-07-11)

cp_hp_v26_scratch finished (90 ep). v25 vs v26 = identical recipe, identical
clean scope, identical split (split_seed 0); the ONLY variable is
initialization (synth pretrain vs scratch):

| | v25 (pretrain init) | v26 (scratch) | delta |
|---|---|---|---|
| VAL best micro-F1 | **0.7066** @ep50 | 0.6183 @ep55 | **+8.8pp** |
| TEST pooled F1 | **0.7341** | 0.6517 | **+8.2pp** |
| VAL pooled P / R | 78.7 / 64.1 | 73.7 / 53.3 | recall is the win |
| VAL ang | 66.2 deg | 72.6 deg | ~same (both bad) |

Far beyond the +-0.03 seed-noise band, consistent across val AND held-out
test, and the gain is concentrated in RECALL -- exactly what the recessed-CP
synthetic mixture was built to teach. The 11i hypothesis is confirmed end to
end: PXC/recessed failure was a learning problem, and generic recessed-CP
pretraining fixes a large part of it.

Standing recipe from here: **synth-pretrain init is ON by default.** v27
(corpus v3, scratch -- launching now) stays scratch so the data lever is
measured in isolation; if v27 lands as expected, v28 = corpus v3 + pretrain
init combines the two proven levers.

### 11n. v27 corpus-v3 verdict: THE DATA LEVER LANDS -- val 0.8748 / test 0.8300 (2026-07-12)

First training on wscad_corpus_v3 (1028 CAD-labelled TB parts; 838 train /
106 val / 93 test after geometry grouping; scratch init, augment 0 after the
OOM night -- see crash logs; 14.1h total incl. 2h hierarchy prebuild).

| | v26 (136 parts) | v27 (1028 parts) | delta |
|---|---|---|---|
| VAL best micro-F1 | 0.6183 | **0.8748** @ep30 | **+25.7pp** |
| TEST pooled F1 | 0.6517 | **0.8300** (P 82.7 / R 83.3) | **+17.8pp** |
| VAL ang | 72.6 deg | **37.8 deg** | direction fixed itself |

- The 11c learning-curve prediction (~1300 parts -> ~0.90) is tracking almost
  exactly. Curve during the run: ep5 0.652 -> ep15 0.822 -> ep30 0.875 ->
  plateau ~0.87.
- Train/val gap (92.1 vs 87.5 pooled) is modest: no pose overfit at augment 0
  with 1028 unique parts -- the aug-0 decision is validated.
- **Direction head healed by data alone**: 73->29-38 deg without touching
  w_dir. The planned two-phase/decoupling intervention is CANCELLED.
- **Threshold sweep verdict: already calibrated.** 7-point sweep on the v27
  val set: 0.30 (the default) is optimal at 87.5%; 0.35 ties with P 90.5 /
  R 84.6 (the high-precision operating point for robot delivery). Unlike the
  v22-era seeds, no free points from decode tuning.
- PXC per-family rows are no longer meaningful (2 val / 2 test parts after the
  11j scope cut); the PXC story continues via the 9-part CAD/fusion track.
- Acceptance gates: val 0.90 (2.5pp short), test 0.85 (2.0pp short),
  family>=0.80 (wscad: val 87.8 / test 83.3 -- test side 3.3pp over the line
  only pooled; micro per-family 83.3 PASSES).

**Launched v28** (run_hp_v28_ftc3.yaml): v27 recipe + init_from synth
pretrain -- the two proven levers combined (data +25.7, pretrain +8.8 on the
small corpus; expect diminishing but positive interaction). Remaining ladder
to 0.90: v28, then 3-seed consensus ensemble on the best recipe.

### 11o. *** 90% REACHED *** v28 = corpus v3 + synth pretrain (2026-07-12)

**Final numbers (checkpoint `checkpoints/cp_hp_v28_ftc3_best.ckpt`, best
@ep45 of 60; decode threshold 0.25 chosen on VAL ONLY, then applied to the
untouched test set in a single shot):**

| set | thr | TP | FP | FN | P | R | micro-F1 |
|---|---|---|---|---|---|---|---|
| VAL (106 parts) | 0.25 | 584 | 63 | 67 | 90.3% | 89.7% | **90.0%** |
| TEST (93 parts, never tuned) | 0.25 | 553 | 55 | 65 | 91.0% | 89.5% | **90.2%** |

Default-threshold (0.30) reference: val 0.8978 / test 0.8968 -- the gates were
passed even before decode tuning. loc 0.91-0.95 mm, ang 29-33 deg.

**Acceptance criteria (P0, user-defined):**
- val micro-F1 >= 0.90: PASS (0.900)
- test >= 0.85: PASS (0.902, +5.2pp margin)
- per-family >= 0.80: wscaduniverse PASS (90.0 val / 90.0 test @0.30);
  PXC has only 2 val + 2 test parts post-scope-cut -- statistically empty,
  tracked on the 9-part CAD/fusion lane instead
- fusion not worse than CAD-only: OPEN (CAD FP-suppression lane, separate)

**The recipe that got here** (each step measured, see 11i-11n):
1. hierpoint backbone (11c: +6pp over knngraph)
2. corpus_v2 label fixes -> CAD-direct **corpus v3: 1028 parts** (+25.7pp,
   the single biggest lever, exactly on the 11c learning curve)
3. scope hygiene: pxc_out_of_scope.txt (12 non-TB parts out)
4. synthetic pretrain with realistic recess mixture (+8.8pp small-corpus,
   +2.3pp on corpus v3; also healed recall and direction)
5. per-run decode threshold sweep on val (0.30 -> 0.25, +0.2pp val, and the
   honest test number ROSE to 90.2)
6. augment 0 at 1028 parts (validated: train/val gap stayed modest)

Direction/location quality came along for free: ang 73 deg (v22 era) ->
29-33 deg, loc ~1.4 -> ~0.91 mm. elapsed 14.0h/run on the T1200.

**Remaining (beyond the ML gate):** CAD FP-suppression + fusion do-no-harm
re-measurement (11k lane), PXC story via vendor STEPs, optional consensus
ensemble if a production margin above 0.90 is wanted.

### 11p. Accuracy-90 P0: error mining + label-ceiling audit (2026-07-12)

Goal shifted to accuracy 0.90 (= F1 ~0.947): the val error budget must halve
(130 -> ~68). Two P0 diagnostics on v28 @ thr 0.25:

**(1) Error mining (error_mine.py, 106 val parts):** errors are DIFFUSE --
worst part has 9, 38% of parts are error-free. One family stands out:

| family | parts | F1 | signature |
|---|---|---|---|
| UK (classic screw) | 4 | **59.5%** | 13/24 GT missed -- recall hole |
| UT | 18 | 84.9% | FP-heavy (16 FP) |
| rest | 80 | 92.0% | proportional |

UK deep screw-funnel entries are the one systematic blind spot (worst parts
3005099/3005837/3005840, all UK). Actionable: UK-style deep-funnel geometry
into the synth generator; check UK share in corpus v4.

**(2) Label-ceiling audit (diag_encoding, corpus-v3 val subset, 103 parts):**
GT-decode ceiling **F1 = 0.994** (min_votes=1; P 1.000, R 0.988, 8 FN / 642).
Labels are essentially clean -- label noise explains at most ~0.6pp. Deep
relabeling is NOT a worthwhile lever; the 130 errors are model errors.

Roadmap consequence: accuracy-90 rides on (a) corpus v4 (building: 2012-file
pool incl. the audited 4th batch: 921 kept / 79 distribution-manifold deleted
+ guarded), (b) UK-targeted synth pretrain v2, (c) multi-seed consensus.

### 11q. CAD FP-suppression sprint: 32.5 -> 57.8 F1 on the 9-part PXC eval (2026-07-13)

Filter-stack ablation (each step measured with cad_eval, human GT, Hungarian
5mm), run WHILE v29 trains (CPU-only, zero GPU touch):

| config | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| baseline --auto --check-dirs (11k) | 13 | 49 | 5 | 21.0 | 72.2 | 32.5 |
| + --dir-consensus (existing, was off) | 13 | 38 | 5 | 25.5 | 72.2 | 37.7 |
| + --drop-corner 0.10 (existing, was off) | 12 | 19 | 6 | 38.7 | 66.7 | 49.0 |
| + --min-depth 2.0 + --deep-axis (NEW) | 13 | 14 | 5 | 48.1 | 72.2 | **57.8** |
| + --auto-multi + --rect-slots (NEW) | 11 | 22 | 7 | 33.3 | 61.1 | 43.1 REJECTED |

**Locked CAD deploy config:** `--auto --check-dirs --dir-consensus
--drop-corner 0.10 --min-depth 2.0 --deep-axis` -> P 48.1 / R 72.2 / F1 57.8.
PT-series: 4/4 parts PERFECT (1.00) -- the bridge-shaft FPs are fully dead.

New code (step_openings.py): `depth_mm` recorded per opening (deepest member
extent along axis); `--min-depth` FP filter (shallow LED windows / marking
pockets die; 3.0mm measured TOO aggressive, kills real bores); `--deep-axis`
(axis from deepest member, not member-count majority); `auto_radius_bands()`
+ `--auto-multi` multi-cluster radius bands.

**Open items (measured, not speculative):**
1. Fuse twins (3036550/63) ang=90 SURVIVES --deep-axis: the fuse cartridge
   shaft is genuinely deeper than the wire bore, so "deepest member" picks the
   wrong axis there. Next idea: extend the mesh-clearance probe to VETO axes
   blocked in both signs (it currently only flips sign).
2. NEOZED 3048357 still 0/2: --auto-multi admits the 6.0mm screw cluster but
   the openings still don't land within 5mm of GT -- needs a per-part look
   (GLB), not more band tuning.
3. rect-slot 3031238 still 0/2: --rect-slots as-is sprays FPs on PT/PTME
   (2-5 each) while NOT recovering the slot; the planar-pocket path needs a
   real fix before it can ship.
4. Fusion do-no-harm re-measure vs the new 57.8 CAD baseline: queued for
   after v29 (needs ML preds on the 9 parts).

### 11r. Full-codebase audit + Phase-1 fixes (2026-07-13)

Three read-only auditors (training/model, data/label, measurement/tooling) went
over the codebase while v29 trained. 39 findings; the load-bearing ones and what
was done about them:

**FIXED (Phase 1, all 43 tests green, zero touch to the live v29 process):**

| # | Defect | Fix |
|---|---|---|
| A1 | `write_labeled_corpus()` **cannot accept** `min_depth`/`deep_axis`/`auto_multi` -- the very FP/direction filters that took CAD from 32.5 to 57.8 (11q) were unusable at LABEL time, so corpus v4 has the errors the deploy path deletes baked into its GT (incl. the wrong axis on fuse blocks) | params added + forwarded from `main()`; corpus v5 can now be built with the deploy config |
| C1 | kNN graphs held as **int64** -- ~13GB resident on corpus v4, the direct driver of three OOM kills | `_knn_graph` returns **int32** (verified bit-identical results; widened to int64 after the device copy, as hierpoint's collate already did). Residency halves to ~6.5GB |
| C2 | The multi-hour pooling-hierarchy prebuild **has a disk cache** -- gated behind `--knn-cache-dir`, which v27/v28/v29 never passed, so every launch AND every resume paid it again (6.2h on the v29 resume alone) | flag renamed `--prep-cache-dir` (old name kept as alias), help text states what it actually gates, loud warning when hierpoint runs without it, and cache writes are skipped below 3GB free so a cache can never fill the disk out from under the run |
| C3 | `--resume` **re-derives** the split from the live corpus; `three_way_split`'s balance search depends on corpus contents, so adding/dropping parts can move a trained part into val -- silently (the config fingerprint sees nothing) | resume now **replays the frozen manifest** (which was already being written but never read back), logs `RESUME SPLIT DRIFT` with counts when a fresh derivation disagrees, and warns when no manifest exists. 3 new tests (`tests/test_resume_split.py`) |
| D1 | OOM/CPU fallback divided the loss by `accum` but **not by the batch's `total_w`** (the fused GPU path does) -> a gradient ~batch-size too large, injected exactly when memory is already tight | `_accumulate(..., weight_div=)`; both fallback loops pass the batch's summed graph weight |
| D2 | Label-time face merge (4.0mm ceiling) != decode-time NMS (5.0mm) despite a docstring claiming one rule -> two real holes 4-5mm apart are labelled as 2 CPs but can only ever decode as 1: **an FN no model can avoid** | single source of truth `cp_targets.NMS_CEILING_MM` / `merge_radius_mm()`, imported by `step_openings` |

**MEASURED, NOT YET FIXED (the honest state of the headline numbers):**
- **A2 label-convention split:** CAD-corpus CPs sit at the bore's centre of mass
  (mid-depth, `step_openings.py:132`), human GT marks the mouth. Measured
  CP-to-nearest-surface distance: **human 3.05mm mean vs CAD-corpus 0.62mm**. The
  model is being taught two conventions at once; the 5mm match radius hides it in
  F1 but it puts a floor under loc error. Fix belongs in corpus v5.
- **B1:** the F1 gates are **blind to direction** -- an all-180-degrees-flipped model
  scores F1=1.0 and passes every acceptance gate (angle is reported, never gated).
- **B2:** the certified operating point (thr 0.25) is NOT what a raw checkpoint
  deploys (`decode.heatmap_thresh` = the training default 0.30).
- **B3:** the 5mm match radius exceeds CP pitch on ~20mm parts, so 0.900/0.902 are
  somewhat optimistic on exactly the dense small parts that dominate the corpus.
- Plus: `is_patch_part` gates the inference path on a vendor NAME prefix (a customer
  mesh silently takes the untested uniform-subsample path); `--exclude-parts-file`
  matches full strings, so the 79-part re-download guard would not fire on
  `wscaduniverse_<id>` names; `error_mine`'s family map drops ~75% of parts into "?".

Phase 2 (after v29): direction gate, threshold unification, scale-adaptive match
radius + honest re-measure of 0.900/0.902, and the 5 missing tests (starting with
`metrics.keypoint_report`, which has **no** unit test today).
Phase 3: corpus v5 (deploy filters + mouth convention) -> v30.

### 11s. The corpus-v5 label ideas: MEASURED and REJECTED (2026-07-13)

Before rebuilding the corpus around the "better labels" hypothesis of 11r, each
proposed label change was scored on the 9-part HUMAN-GT set (cad_eval, Hungarian
5mm). All three failed:

| label config | TP | FP | FN | F1 | ang | verdict |
|---|---|---|---|---|---|---|
| `--auto --check-dirs` (the v4 recipe) | 13 | 49 | 5 | **32.5%** | 38.6 | baseline |
| `+ --entry-at-mouth` (11r's A2 "fix") | 5 | 57 | 13 | **12.5%** | 90.0 | **HARMFUL** |
| `+ --deep-axis` | 13 | 49 | 5 | 32.5% | 38.6 | **NO EFFECT** (bit-identical) |
| `+ min-depth / drop-corner / dir-consensus` | -- | -- | -- | -- | -- | **POISON as labels** |

1. **`--entry-at-mouth` is wrong.** The A2 diagnosis (CAD CPs sit at bore
   mid-depth, human GT marks the mouth) is real, but pushing the point out by
   `depth/2` overshoots: on a MERGED opening the centre is a mean of member
   centres while `depth` is the deepest member's, so the point lands past the
   mouth, outside the part -- TP 13 -> 5 against human GT. A correct fix needs a
   ray-cast onto the surface, not a blind push. Flag kept, OFF, documented.
2. **`--deep-axis` does nothing measurable.** Its inclusion in the 11q "locked
   config" was never attributable -- it was added together with `--min-depth`, and
   the angle error was unchanged (38.6 both ways).
3. **The FP-suppression filters cannot be label rules.** They were tuned on 9
   PT/ST/fuse parts. Applied to a 110-part family-spanning sample they delete
   **73% of all CPs** (716 -> 193) and leave **25 of 92 parts with ZERO CPs** --
   and a terminal block always has at least two wire entries, so those are real
   terminals being cut. Per-filter on a 40-part sample: min-depth -27%,
   dir-consensus -22%, drop-corner similar.

**Correction to 11q:** the CAD-direct 32.5 -> 57.8 result is REAL but
FAMILY-LOCAL (PT/ST/fuse). It is a prediction-time config for those families, not
a general detector improvement, and NOT a labelling config. The honest CAD-direct
number on an unseen family remains unmeasured.

**Consequence:** corpus v5 = the v4 recipe + the unified merge/NMS ceiling (11r
D2, which removes decode-unreachable label pairs). No other label change survives
evidence. v30 = v29's experiment (1741 parts + synth-pretrain init) on it, with
the Phase-1 fixes (int32 graphs, process-pool prep, resume manifest guard, OOM
gradient fix).

### 11t. Label merge vs decode NMS: the audit's "unify them" fix was WRONG (2026-07-13)

11r/D2 flagged that the label-time face merge (4.0mm ceiling) and the decode-time
NMS (5.0mm) disagree, so two real holes 4-5mm apart on a big part are labelled as
2 CPs but can only ever decode as 1 -- an unavoidable FN. The diagnosis is right;
the fix I applied (coarsen the LABELS to 5.0) is not. Measured on the HUMAN ground
truth:

**37 pairs of REAL terminals, on 8 parts with >100mm bodies, sit 4-5mm apart.**

Merging those at label time would (a) teach the model that two wire entries are one
point -- the robot would be handed one of them, a genuine product regression -- and
(b) silently reclassify those unavoidable FNs as TPs, INFLATING F1 without improving
anything, and breaking comparability with the v28 baseline (whose 0.900/0.902 pays
for them honestly).

**Resolution:** labels stay at 4.0mm (truthful, and identical to what v28/v29 were
trained on); decode stays at 5.0mm (its real resolution). Constants are now explicit
and documented (`cp_targets.LABEL_MERGE_CEILING_MM` = 4.0 vs `NMS_CEILING_MM` = 5.0).
The residual FN is a KNOWN, PRICED-IN limitation. The correct fix lives on the decode
side (finer ceiling and/or vote-density-aware clustering) and must be measured -- on
both close-pair recall AND the double-detection FPs it risks -- before it ships.

**Consequence for corpus v5:** it is a faithful rebuild of the v4 recipe (`--auto
--deflection 0.5 --check-dirs --merge-tol 4.0`, all FP filters off -- see 11s), on the
2011-STEP pool (one unmeshable STEP deleted). v30 = v29's experiment on it, so its
numbers ARE directly comparable to v28's 0.900/0.902.

### 11u. A2 (the "labels sit at mid-depth" defect): CLOSED, hypothesis refuted (2026-07-13)

11s rejected `--entry-at-mouth` but blamed the *implementation*, and left the
door open: *"A correct fix needs a ray-cast onto the surface, not a blind push."*
That correct fix is now implemented and measured. It does not help. The premise
itself is wrong, so A2 is closed rather than deferred.

`mouth_coord()` (step_openings.py) takes the mouth from the MEMBER FACES' OWN
outer ends -- a bore/funnel face physically ends at the mouth, so its extreme
along the approach axis IS the surface; no ray-cast needed. It cannot overshoot
(the old push mixed frames: the merged centre is a MEAN of member centres while
`depth` is the DEEPEST member's, so a funnel+bore merge landed *outside* the
part) and it is clamped to the part bbox. 3 regression tests pin this
(`tests/test_cad_geom.py`), including the exact funnel+bore case that broke the
blind push. Suite: 46/46.

Scored on the same 9-part HUMAN-GT set, same scorer (cad_eval, Hungarian, 5mm):

| label config | TP | FP | FN | F1 | ang | verdict |
|---|---|---|---|---|---|---|
| `--auto --check-dirs` (baseline, reproduces 11s exactly) | 13 | 49 | 5 | **32.5%** | 38.6 | -- |
| `+ --entry-at-mouth`, blind push (11s) | 5 | 57 | 13 | 12.5% | 90.0 | HARMFUL |
| `+ --entry-at-mouth`, **correct mouth** (mouth_coord) | 5 | 57 | 13 | **12.5%** | 90.0 | **HARMFUL** |

The fix is verified live, not silently inert: all 62 CPs move, by 4.39mm mean /
12.85mm max. It lands the point on the real surface -- and still halves TP.

**So the mid-depth centre is CLOSER to human GT than the mouth is.** Whatever the
"CAD 0.62mm vs human 3.05mm from nearest surface" measurement of 11r captured, it
is not "human GT marks the mouth": if it were, moving labels to the mouth would
have raised TP, and it lowers it (13 -> 5) no matter how exactly the mouth is
computed. Do not re-open this without new evidence about where human CPs actually
sit; two independent implementations of the same idea both fail the same way.

`--entry-at-mouth` stays OFF by default. `mouth_coord()` is kept (correct,
tested) so that the flag, if ever switched on, is at least not also buggy.
