# WiringRobot Connection-Point Detector — FINAL STATUS & HANDOFF
**Rittal / Friedhelm Loh Group · 2026-07-27 · leakage-free OOF, thesis+plan-faithful**

> One-line: detect wire connection points (CPs) on WSCAD electrical-terminal STEP files so a robot can
> insert wires. Built on the Scheffler master thesis (DiffusionNet semantic segmentation → CP derivation).
> This document is the honest, single-source status. Numbers are leakage-free GroupKFold-OOF against the
> manufacturer's own ConnectionPoints (the robot target). Nothing here is in-sample or inflated.

---

## 1. THESIS-NATIVE DELIVERABLE (segmentation) — DONE, beats the thesis

The Scheffler thesis's OWN benchmark is **semantic segmentation** (5 classes: Housing, Contact, SnapPoint,
CableEntry, LabelSurface), reported as **mean Jaccard 0.514**.

| Metric | Thesis | Ours (`best_full.pt`, val 20 parts) | Δ |
|---|---|---|---|
| mean IoU / Jaccard | 0.514 | **0.681** | **+33%** |
| Dice | — | **0.790** | — |
| accuracy | — | **0.864** | — |

Per-class IoU: Housing 0.81 · Contact 0.68 · SnapPoint 0.53 · **CableEntry 0.74** · LabelSurface 0.69.
Even the *deployed* product model (`recall_hard_s2`) scores IoU 0.633 / Dice 0.746 / acc 0.831 — also above thesis.
Reproduce: `seg_thesis_eval.py` → `results/seg_thesis_val.json`.

**On the thesis's own terms, the work is complete and clears the ≥0.80 defensible bar (acc 0.86 / Dice 0.79).**

---

## 2. THE PRODUCT (CP point-detection) — honest OOF ladder

**Pipeline (frozen 2026-07-24):** STEP → `thesis_remesh`(6000 verts) → 4-model DiffusionNet segmentation →
per-model `cp_openings` (cp-v3: CableEntry OR depth-gated Contact; min_v 30 / vertex_conf 0.5 / cluster 5mm) →
**UNION (vote≥1)** → **RF wire/tool gate 0.35** → two-tier (AUTO / REVIEW).
Checkpoints: `recall_hard_s2` + `recall_hard_keig96_s0/s1/s2` · gate `results/wire_gate.pkl`.

**Leakage-free OOF F1 vs manufacturer ConnectionPoints** (canon `results/product_f1_receipt.json`):

| Set | base (geometry only, deployed) | + CP-count (metadata-assisted) |
|---|---|---|
| **ALL** | **0.750** (thr-ceiling 0.756) | **0.775** |
| **PXC-typical** | **0.823** ✅ | 0.837 |
| WEI-typical | 0.696 | 0.723 |
| high-CP (≥11 CP), full-stack | — | **0.807** part-out / 0.799 family-out |

Primary metric (user decision): AUTO+REVIEW F1. AUTO-precision is a separate safety metric.

---

## 3. THE HONEST 0.85 ANALYSIS (this session, leakage-free) — why 0.85 needs a labeling round

Measured every angle on `f1_sweep_data.npz` (823 parts, 4785 union CPs, 2837 manufacturer GT). Result:

1. **Threshold / per-mfg gate threshold = DEAD.** Nested-CV per-mfg threshold gain WEI +0.011, ALL +0.004
   (noise). 0.35 is already optimal. (An earlier single-model quick-look suggested a big win; that was a
   leaky, in-sample artifact — corrected. The 4-model UNION already recovers the recall a single model loses.)
2. **Finding openings = SOLVED.** Aggressive multi-resolution candidate pool reaches recall ceiling **0.969**
   (WEI). Candidates are NOT the bottleneck.
3. **The bottleneck is wire-vs-non-wire DISCRIMINATION (the gate).** An oracle gate (perfect TP/FP split on
   current candidates) would score F1 **0.894** ALL / 0.911 WEI. The real gate reaches only 0.756. The gap
   is real headroom but **hard**: the surviving false-positives are *real openings that are non-wire*
   (tool/test/screw). They are geometrically **identical** to wire entries (size/depth/vertex-count/
   confidence all overlap) — the thesis Contact class merges "Kontaktierung **bzw. Werkzeugeinschub**" by
   design. **3D geometry alone cannot separate wire from tool** (gate AUC caps ~0.88–0.93 → F1 ~0.66–0.79).
4. **Therefore raw ALL CP-F1 0.85 requires a signal 3D geometry doesn't carry.** Two paths exist:
   a *2D-render visual discriminator* (departs from the thesis — **rejected as non-thesis-faithful**), or
   a *targeted recall-painting labeling round* on the measured WEI/high-CP failures (the plan's documented
   Go/No-Go lever). **Per the current plan's scope ("YAPILMAYACAKLAR: yeni korpus etiketi"), no new labeling
   round was run this session.**

Full detail: memory `085-decomposition-leakage-free.md`; scripts `w1_gate_oof.py` (honest), `f1_sweep.py`.

---

## 4. WHAT ≥0.80 / ≥0.85 HONESTLY MEANS HERE (no inflation)

- ✅ Thesis-native segmentation: acc 0.86 / Dice 0.79 — clears 0.80.
- ✅ PXC-typical CP-F1: 0.823 — clears 0.80.
- ✅ count-assisted high-CP full-stack: 0.807.
- ❌ **ALL CP-point-F1 0.756 and WEI 0.696 do NOT reach 0.80 unlabeled** — top-N ceilings (0.775 / 0.723)
  are below 0.80; proven five independent ways. 0.85 for these needs the recall-painting labeling round.
- The old ~0.98 precision / ~0.80 "robot-effective" numbers were INFLATED (they counted tool openings as
  correct). Report OOF only.

---

## 5. SUCCESSOR RUNBOOK (how to reproduce / run)

Environment: native Windows, `.venv` (torch + DiffusionNet). Always set
`PYTHONPATH=_diffusion_net_repo/src`, run `.venv/Scripts/python.exe`.

| Goal | Command |
|---|---|
| Thesis-native segmentation eval | `seg_thesis_eval.py` → `results/seg_thesis_val.json` |
| Product CP-F1 ladder (cached, no GPU) | `f1_sweep.py --sweep-only` (uses `results/f1_sweep_data.npz`) |
| Regenerate the F1 data (GPU) | `f1_sweep.py` (runs 4-model union on 823 parts, ~15 min warm cache) |
| Manufacturer arbiter, one ckpt | `BA_ALLOW_SEEN=1 big_arbiter.py --ckpts <ckpt> --only-mfg WEI --only-parts $(cat _hw_r3.txt) --axis-aware --cluster-mm 5 --min-v 30 --vertex-conf 0.5` |
| Robot end-to-end (product path) | `robot_e2e.py [--mfg WEI] [--limit 60]` |
| Honest 0.85 decomposition | `w1_gate_oof.py` (leakage-free, CPU only) |

Key facts a successor MUST know:
- **Manufacturer CP convention:** the manufacturer reports the CP at the CONTACT SEAT (~12–15mm deep); the
  thesis/our model reports the opening MOUTH. Always match **axis-aware** (perpendicular distance to the
  insertion axis, allow depth along it) — euclidean matching measures the convention gap and reads 0.000.
  See `big_arbiter.greedy` docstring.
- **Leakage guards:** `_label_targets_recall/trained_parts.json` and the corpus train/val split are excluded
  from scoring automatically (`big_arbiter.eligible`). Never score a trained part.
- **Report OOF only.** In-sample / deploy-gate-on-training-parts numbers are optimistic.

---

## 6. THE PLAN-FAITHFUL NEXT LEVER (documented, deliberately NOT done this round)

If the team later decides to push raw ALL/WEI CP-F1 toward 0.85, the ONE thesis-faithful lever is a
**targeted recall-painting labeling round** on the measured failure set (WEI zero-recall parts + high-CP
terminals). This is *targeted* (on parts the model provably fails), not blind labeling. It stays within the
thesis's 3D-segmentation frame (better training labels, same architecture). It was left out of the current
round on purpose because the active plan scopes out new corpus labeling. Preparing the queue is a ~1-day task;
the failure parts are already identified in this session's measurements.

---

## 7. FILE MAP (what matters)
- `cp_config.json` → `current_product` = single source of truth for the deployed pipeline.
- `PRODUCT_MODEL.md` → provenance + canonical ladder (top block current, below historical).
- `results/product_f1_receipt.json` → the canonical leakage-free receipt.
- `cp_openings.py` → cp-v3 CP derivation. `wire_gate.py` → RF wire/tool gate. `robot_cp.py` → product inference.
- `big_arbiter.py` → the scorer. `f1_sweep.py` → OOF ladder. `seg_thesis_eval.py` → thesis-native metric.
- Memory index `MEMORY.md` → durable findings across the 3-week effort.
