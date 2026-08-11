# Overnight work — 2026-07-18 (autonomous)

You gave a P0-P3 audit and asked me to FBI-review it, do the achievable parts, then self-FBI and
do a second round. Done. The audit was **largely correct** and I acted on it honestly.
Full assessment: `FBI_SELF_REVIEW.md`.

## The single most important correction
Your CP ontology critique was right, and I verified it myself: **CableEntry-only matches the
manufacturer connection count** on the two cleanly-labelled parts (0444048 2=2, 0446017 2=2),
while my old Contact+CableEntry recipe **over-counted (5 vs 2)**. New frozen definition
**`cp-v2-cableentry-primary`**: a CP = one CableEntry opening (physical wire entry); **Contact is
auxiliary evidence, never auto-counted.** The old numbers are superseded.

## Honest metrics now (three, never mixed — `cp_config.json`)
| metric | value | what it really is |
|---|---|---|
| **1 semantic** (locked-11) | acc 0.858 / IoU 0.680 | REAL historical benchmark |
| **2 region-derived CP consistency** (cp-v2) | val F1 0.989, test 0.872 | dev only; GT auto from REGION labels; test was diagnostically touched → NOT held-out; scores only 82/102 parts |
| **3 human physical CP** | **NOT MEASURED** | no human point+direction labels exist |

No CP number is "human-confirmed", "held-out", or "final" anymore.

## Round 1 — audit items done
- **CP ontology → CableEntry-primary**, manufacturer-validated (`cp_ontology_table.py`).
- **3 metrics named separately** across report + memory + receipts.
- **GT/prediction separated**: GT is the fixed definition (byte-stable); min_v + confidence mask
  apply to PREDICTION only. One evaluator `cp_evaluate.py` → JSON receipts w/ SHAs (`results/cp_eval/`).
- **Fake human-confirm CLOSED**: `cp_review_package.py` emits PENDING (unsigned); `cp_seal_score.py`
  REFUSES to score until every CP is decided + seal signed + SHA matches. Auto-confirmed-as-human = **0**.
- **One split manifest** (`split_manifest.json`, roles; the 11 marked BURNED for CP-final),
  **one config** (`cp_config.json`), **promotion gate** (`promotion_gate.json`: P/R/Jaccard≥0.90,
  ≤5mm, ≤15°, 100% coverage), **provenance map** (`PROVENANCE.md`).
- **Tests** (`tests/test_cp_openings.py`, 7 green; full suite 66 green).

## Round 2 — my own FBI on the round-1 state
- **Coverage honesty (real catch):** cp-v2 only scores **82/102** parts (val 16/20, test 9/11).
  19 parts label connections as *Contact* not CableEntry (+1 empty). The receipts now say so — the
  earlier F1s silently hid this (the audit's selective-coverage risk). **This is the biggest open
  GT problem: ~20% of the corpus is labelled inconsistently.**
- **Direction sanity (never checked):** 99.0% outward, 97.9% axis-aligned — sound.
- **OBJ↔STEP frame PROVEN identity** (`cp_transform_receipt.py`): 102/102, worst center residual
  0.312mm. Human clicks in STEP will register with model CPs within 0.3mm.
- **Geometric corroboration inconclusive:** step_openings finds screw cylinders, not cable funnels.
- **Untouched benchmark candidates:** `benchmark_candidates.json` — 40 parts, denylisted vs the 102
  (ids + STEP SHAs), SHA-deduped, 39 families, overlap=0. UNLABELLED.

## What needs YOU (I could not do autonomously)
1. **Human point+direction CP labels** in STEP frame → the real metric #3. Tool is ready and now
   defaults to PENDING; sign the seal to score. Start with `benchmark_candidates.json` (untouched).
2. **Resolve the 20 Contact-only / gt_incomplete parts** — decide: re-label their connections as
   CableEntry, or define Contact-as-wire-entry for those families. This is the top GT issue.
3. **Catalog lookup** for the other 98 manufacturer counts (I had only the 4 you gave).
4. **Git**: ~184 uncommitted incl. your deletions — I did NOT touch git (deletions need your
   one-by-one review; committing is yours). New files listed in `PROVENANCE.md`.

## Strategy FBI (later session) — faithfulness to thesis + JSON + cp-v2  (`STRATEGY_FBI.md`)
- **Biggest weakness found & fixed:** the `Desktop\JSON` corpus (479 meshes, **11,927 real
  manufacturer CPs with point+direction**) was sidelined and metric #3 called "unmeasurable".
  It IS the metric-#3 resource. Validated cp-v2's conventions against it (`json_cp_convention.py`:
  InsertDirection **100% axis-aligned, 98.3% outward** = cp-v2's model) and built the in-domain
  benchmark (`json_cp_benchmark.py`: SHA-deduped 341/74/64 split, point+direction scorer 5mm/15°).
  Metric #3 is now MEASURABLE (learned detector = next step; in-domain known-feasible).
- **Real nuance:** manufacturer CP point sits ~4.7mm OFF-surface (a standoff), vs cp-v2's
  on-surface opening midpoint. The JSON scorer targets the manufacturer point (handled); flagged
  for the robot target.
- **Thesis reconciliation:** the thesis localises BOTH Contact+CableEntry; cp-v2 counts CableEntry
  only — faithful to the user's frozen cable-entry task + manufacturer 1-per-terminal count. Contact
  localisation stays available for screw actuation.
- **Scale fork (F4):** cp-v2 is Scheffler-segmentation-bound (102 parts). The full WSCAD-STEP target
  needs either more hand-labelled terminal blocks OR reviving JSON-train + remesh-transfer (dormant).
- Bug sweep clean (gt_min safe, vertex_conf=0.9 re-validated, test-peek guard added, tests green).

## One-command reproduce
`PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_evaluate.py --split val` (or test_locked)
→ receipt in `results/cp_eval/`. Everything is config-driven from `cp_config.json`.
