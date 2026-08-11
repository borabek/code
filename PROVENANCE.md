# Data & result provenance (2026-07-18) — keep these SEPARATE, never mix scores

The audit's core discipline: every number carries its provenance; sets with different origins
are never pooled. This file is the map.

## A. Scheffler-Bründl human-labelled corpus  (`wscad_corpus_scheffler_exact/`)
- Human semantic segmentation (Housing/Contact/SnapPoint/CableEntry/LabelSurface) matched to the
  exact WSCAD STEP. Split frozen in `split_manifest.json`: 71 train / 20 val / 11 test_locked.
- **Roles:** train+val = development. test_locked = historical SEMANTIC benchmark (metric #1) and
  CP regression/dev — **BURNED for any CP-final claim** (CP rules diagnosed on it; v31 trained on
  all 11, v28 on 6).
- CP definition applied to it: `cp-v2-cableentry-primary` (`cp_config.json`).

## B. Manufacturer connection counts  (external, product pages / Phoenix Contact)
- Ground-truth CONNECTION COUNTS for validating the CP ontology. Known for 4/102 parts so far
  (`cp_ontology_table.py` MFR dict). The other 98 need catalog lookup (flagged needs_human).
- Used ONLY to validate counts (CableEntry-primary), never as CP point labels.

## C. Desktop\JSON manufacturer corpus  (`C:/Users/DE00024082/Desktop/JSON`, 479 parts)
- 11,927 manufacturer CPs (Point + InsertDirection + terminal Name). These are A-B / different
  part families — **zero ID overlap** with the Scheffler corpus (verified). Defines the CP
  CONVENTION (1 CP/terminal at the wire-entry opening) but is NOT a label source for Scheffler
  parts. Separate provenance; never pooled with Scheffler scores.

## D. Legacy pseudo-label pipeline  (v22/v25/v26/v28/v31, wscad corpus v2-v8)
- CAD-pseudo-label CP heatmap era. PROVEN unreliable as a target (CAD vs human GT F1~0.46; see
  memory `ground-truth-is-the-ceiling`). **Historical only.** v28/v31 checkpoints saw test_locked
  parts in training -> must NOT be used for any human-CP final claim.

## E. `check/` sets (e.g. wscad_labeled_5)
- Passed automated QA; awaiting human signoff. Each part takes EITHER train OR benchmark role,
  never both.

## Metric → provenance binding (never violate)
- semantic (metric #1)      <- A.test_locked, single-shot. Real.
- region-derived CP (metric #2) <- A, GT auto-derived from A's region labels. Development.
- human physical CP (metric #3) <- a NEW untouched set, human point+direction. NOT YET EXISTING.
- ontology validation       <- B (+ C convention). Counts only.

## Reproducibility (git NOT touched autonomously)
~184 uncommitted entries incl. user deletions. Committing/deleting is left to the user (deletions
must be reviewed one-by-one; committing is outward-facing). New this session (untracked, safe to
keep): cp_config.json, cp_openings.py, cp_evaluate.py, cp_ontology_table.py, split_manifest.py,
cp_review_package.py, cp_seal_score.py, cp_review_glb.py, contact_sheet.py, promotion_gate.json,
FBI_SELF_REVIEW.md, PROVENANCE.md, CP_COVERAGE_REPORT.md, tests/test_cp_openings.py. Run recipe:
`PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_evaluate.py --split val|test_locked`.
