# CP model card (2026-07-18)

The packaged, hash-pinned description of the current CP pipeline. Do NOT treat a bare checkpoint
as production — it is only valid with THIS config + threshold + definition.

## Target scope (frozen)
The target is the **physical wire-entry opening** (a hole/entry point), NOT a logical electrical
terminal. CP = one **CableEntry** connected component (cp-v2). Contact is auxiliary. This matches
the manufacturer connection count on cleanly-labelled parts.

## Artifacts + hashes
| item | value |
|---|---|
| checkpoint | `results/scheffler_semantic/refit91.pt` (sha16 `33d1aa2070694d02`) |
| CP config / definition | `cp_config.json` (sha16 `add089827176dc16`), version `cp-v2-cableentry-primary` |
| derivation | `cp_openings.py` (sha16 `f0735519d1ad8792`) |
| evaluator | `cp_evaluate.py` (sha16 `95db68e7d4535587`) |
| split manifest | `split_manifest.json` (71/20/11; the 11 BURNED for CP-final) |

## Prediction post-processing (PREDICTION only, tuned on val)
- min 20 vertices per component; **per-vertex confidence mask 0.9** (re-validated under cp-v2:
  0.8→F1 0.967, 0.9→0.989, 0.95→0.923).
- Threshold selected on val ONLY; test_locked guarded by `--allow-locked`.

## Results (region-derived CP consistency, cp-v2; NOT human-CP, NOT untouched held-out)
| radius | val F1 | val P/R |
|---|---|---|
| 5 mm (headline) | 0.989 | 0.978 / 1.000 |
| **3 mm (strict)** | **0.879** | 0.870 / 0.889 |
The 5→3 mm drop is position sloppiness — report both. Coverage: scores 82/102 parts (20 gt_incomplete).

## Known failure modes (`results/cp_failure_set.json`)
- **bad_direction (14) = the #1 error**: position matches GT but approach angle > 30°. Direction is
  the weak link, not position.
- missed_cp 7, spurious_fp 6, wrong_surface 2, duplicate/side_rail 0.
- **Family blind spot: 0790xxx** (0790446/462/530/543/491) — systematic missed_cp + spurious_fp + bad_direction.

## Direction gate (per audit P2)
A CP must pass BOTH position (≤5 mm) AND direction (≤15°). High F1 with high angle error is NOT a
PASS. Direction is currently the failing axis — do not report a CP "PASS" on position alone.

## Rules (frozen)
- No "improved" claim without visual QA (`cp_qa_viz.py`, colour standard GREEN=GT/BLUE=TP/RED=FP/ORANGE=FN).
- Final promotion only against `promotion_gate.json` on a NEW human-sealed set (P/R/Jaccard ≥ 0.90, ≤5 mm, ≤15°, 100% cov).
- Provenance separation per `PROVENANCE.md`; WSCAD inventory in `wscad_master_manifest.json`.
