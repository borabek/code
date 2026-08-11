# CP detection — honest status (2026-07-18, post-audit)

Three DISTINCT metrics, never conflated (see `cp_config.json` metric_names):

| # | metric | what it is | status |
|---|--------|-----------|--------|
| 1 | **semantic segmentation** | per-vertex 5-class accuracy on the locked-11 | REAL historical benchmark |
| 2 | **region-derived CP consistency** | CP-F1 where GT is AUTO-derived from human REGION labels | development diagnostic only |
| 3 | **human physical CP** | CP-F1 vs human-clicked point+direction in STEP frame | **NOT YET MEASURED** (no human labels) |

## Data / split (frozen — `split_manifest.json`)
- 71 train + 20 val = **development**. 11 = **historical semantic benchmark + CP regression/dev**.
- The 11 are **BURNED for any CP-final claim**: CP rules were diagnosed on them, and v31 saw all
  11 in training / v28 saw 6. A real CP final needs a NEW untouched set (not yet built/labelled).

## CP definition (frozen — `cp_config.json`, version `cp-v2-cableentry-primary`)
A CP is one **CableEntry** connected component = the physical wire-entry opening. **Contact is
auxiliary evidence, never auto-counted.** Validated against manufacturer connection counts
(`cp_ontology_table.py`): on the two cleanly-labelled parts with a known count, CableEntry-only
matches exactly (0444048 2=2, 0446017 2=2); the earlier Contact+CableEntry recipe over-counted
(5 vs 2) and is superseded (kept as `cp-v1` for provenance only). Parts with 0 CableEntry but a
positive manufacturer count are **gt_incomplete** (0271017 mfr 8→0; 3209635 mfr 3→0) — the human
segmentation misses those connections; they are flagged, not silently back-filled from Contact.
Manufacturer counts known for 4/102; the other 98 need catalog lookup (flagged needs_human).

## Results (honest)
**(1) Semantic segmentation, LOCKED-11 (single shot):** accuracy **0.8581**, mean Dice 0.8050,
mean IoU **0.6797** (beats the thesis 0.51). This is the one defensible current result.

**(2) Region-derived CP consistency (`cp_evaluate.py`, cp-v2, GT auto from region labels):**
- val (development): F1 **0.989** (P 0.978, R 1.000, Jac 0.978), position median 0.59 mm.
- test_locked: F1 **0.872** (P 0.895, R 0.850, Jac 0.773), position median 0.64 mm.
- **Caveats (do not overstate):** GT is auto-derived from human REGION labels, NOT human CP
  clicks; test_locked was DIAGNOSTICALLY TOUCHED (rules changed after looking at its parts), so
  it is **not** untouched held-out. GT is byte-stable across prediction knobs (GT uses the fixed
  definition; min_v/confidence apply to PREDICTION only). Receipts: `results/cp_eval/*.json`.

**(3) Human physical CP: NOT MEASURED.** No human point+direction labels exist. `cp_review_package.py`
emits PENDING candidates; `cp_seal_score.py` REFUSES to score until every CP is decided and the
seal is signed (auto-confirmed-as-human count = 0 by design).

## Prediction post-processing (PREDICTION only, never GT)
Per-vertex confidence mask (class prob < 0.9 → Housing before connected-components: erodes low-conf
FP blobs and splits merged openings) + min 20 vertices. Frozen in `cp_config.json`.

## Visualisation — shows BOTH features (thesis), metric counts CableEntry (manufacturer)
Two separable concerns, deliberately not conflated:
- **CP-COUNT metric** = CableEntry only (cp-v2, manufacturer-validated). This is what `cp_evaluate`
  scores. It is NOT changed by the viz.
- **Review VISUALISATION** shows BOTH connection features per the thesis (§5.3.6 localises
  Kontaktierung AND Kabeleinfuehrung): **green ball = CableEntry (the CP), blue ball = Contact
  (auxiliary feature / screw-clamp).** So a part whose connections are labelled Contact (the 20
  gt_incomplete parts, e.g. 2002-2941) is NO LONGER empty — it shows its blue Contact features for
  human review. `_scheffler_viz/` (semantic 5-class + `_CPREVIEW_*` both-feature markers) and
  `_cp_confirm/` (numbered PNG + colour-by-class review GLB + PENDING confirm JSON). Review-GLB ball
  radius auto-shrinks to 0.38× the nearest gap so markers never overlap.

## Round-2 validations (2026-07-18, autonomous)
- **Coverage honesty (was hidden):** cp-v2 scores only the CableEntry-labelled parts — **82/102**
  (val 16/20, test 9/11). The other 20 are gt_incomplete (19 label their connections as *Contact*
  not CableEntry; 1 has no connection label). The receipts now report this; the F1s above are over
  the scored subset, NOT the whole corpus. This is a GT labelling inconsistency, not a definition
  bug — resolving it needs re-labelling (human).
- **Direction sanity (never checked before):** 99.0% of CableEntry CP directions point OUTWARD,
  97.9% are axis-aligned. The direction head is sound for CableEntry openings.
- **OBJ↔STEP frame PROVEN identity** (`cp_transform_receipt.py` → `results/cp_transform_receipt.json`):
  all 102 parts same_frame, worst center residual **0.312 mm** (extent 1.58 mm = remesh envelope).
  So OBJ-frame CPs are valid STEP-frame targets within 0.3 mm — human clicks in STEP will register.
- **Geometric corroboration inconclusive:** only 2/20 CableEntry CPs sit near a step_openings
  cylinder — because that detector finds screw/clamp holes (Contact side), not cable funnels.
  Different features; neither confirms nor refutes cp-v2.
- **Untouched benchmark candidates ready** (`cp_benchmark_select.py` → `benchmark_candidates.json`):
  40 parts from 3535 (after denylisting the 102 Scheffler ids + STEP SHAs, SHA-dedup), 39 families,
  overlap=0 asserted. UNLABELLED — human point+direction labelling is the next step.

## Open (needs human / next)
1. Human point+direction labels in STEP frame -> the real metric (3). Tool ready, defaults pending.
2. New untouched CP benchmark: SHA-dedup select from all_wscad_stp, denylist vs train history.
3. Catalog lookup for the remaining 98 manufacturer counts.
4. Promotion gate frozen in `promotion_gate.json` (P/R/Jaccard ≥ 0.90, ≤5 mm, ≤15°, 100% coverage).
