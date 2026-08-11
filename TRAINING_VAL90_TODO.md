# Training Val-90 TODO

> SUPERSEDED (2026-07-15): Yeni hedef Desktop JSON manufacturer label'lariyla
> unseen WSCAD STEP uzerinde ML-only CP bulmaktir. Bu eski plan WSCAD/CAD
> pseudo-label akisini merkeze aldigi icin uygulanmayacak. Guncel otoritatif plan:
> TRAINING_JSON_WSCAD_90_TODO.md.

Goal: raise validation-side CP metrics (`val_micro_f1`, F1/accuracy-like values)
toward 90% without fooling ourselves with a tiny or over-tuned validation split.

Current workspace baseline:

- Best current honest mesh-ML result: `cp_hp_v22_best.ckpt`.
- Raw checkpoint history: `val_micro_f1=0.6116` at epoch 70.
- Decode tuning note in `RESULTS.md`: no-TTA `thr=0.15`, `votes=1` gives
  `val_micro_f1=0.644`, P=66.8%, R=62.2%.
- Active run: `cp_hp_v22_seed1` appears to be running from `run_night_queue.ps1`.
- Main blocker seen in logs: PXC/rect-slot family is near-zero on val/test, while
  `wscaduniverse` train can already hit about 90% F1. General 90% needs more
  representative data and labels, not another blind hyperparameter sweep.

## P0. The 90% Push

These are the three tasks that matter most. Do these before spending more time
on ordinary LR/threshold experiments.

- [ ] **PXC eval'i buyut:** expand the PXC/CAD eval from the current tiny 3-part
  set, but only with non-empty geometry-valid STEP files. The 18 WSCAD lookup
  candidates were reported empty by the user on 2026-07-10, so do not count them
  unless a sanity check proves they contain real geometry.
- [ ] **10-15 rect-slot parcayi saglam label'la:** build a verified rect-slot
  subset with STEP, human GT, visual QA, and per-part TP/FP/FN notes.
- [ ] **Balanced `hp_v24` train et:** after the PXC/rect-slot subset exists,
  train `hp_v24_balanced_pxc` from the v22 recipe with PXC/rect-slot oversampling
  or loss weighting.

Acceptance for moving past P0:

- [ ] PXC eval has at least 10 geometry-valid parts, preferably from official
  Phoenix Contact/original CAD sources if WSCAD remains empty.
- [ ] Rect-slot subset has at least 10 verified positive examples and some hard
  negative clutter examples.
- [ ] `hp_v24_balanced_pxc` improves PXC recall without dropping `wscaduniverse`
  F1 by more than 3 percentage points.
- [ ] General val micro-F1 improves beyond the hp_v22 tuned baseline (`0.644`),
  and per-family PXC F1 is no longer near zero.

## 0. Guardrails

- [ ] Do not promote `last.ckpt`; only promote `best.ckpt` or an exported
  checkpoint with embedded decode thresholds.
- [ ] Keep an untouched test report for every candidate. Val can drive decisions;
  test is for final confirmation only.
- [ ] Treat 90% on the current 22-part val as a weak signal. Require multi-seed
  and per-family evidence before calling it real.
- [ ] Keep TTA off for `hierpoint`; `tta_hp_v22.log` shows it hurts this backbone.

## 1. Freeze the current baseline

- [ ] Let the active `cp_hp_v22_seed1` run finish, then run seed2.
- [ ] Summarize the three seed histories:

```powershell
.venv\Scripts\python.exe summarize_cp_runs.py `
  --pattern checkpoints\cp_hp_v22_history.json `
  --pattern checkpoints\cp_hp_v22_seed1_history.json `
  --pattern checkpoints\cp_hp_v22_seed2_history.json `
  --baseline 0.644 `
  --max-gap 0.03
```

- [ ] For every seed, run the same no-TTA decode sweep including low thresholds
  (`0.10,0.12,0.15,0.18,0.20,0.25,0.30`) and record the real best operating point.
- [ ] Acceptance before new experiments: 3-seed mean val micro-F1 should be close
  to the seed0 tuned value (`~0.64`) and std should be <= 0.02. If variance is
  larger, do not trust single-run gains.

## 2. Fix the evaluation set before chasing 90

- [ ] Audit how many PXC parts have both human GT and STEP available now:

```powershell
Get-ChildItem "C:\Users\DE00024082\Desktop\JSON" -Filter "PXC*.json" | Measure-Object
Get-ChildItem all_connectors -Filter "*PXC*.stp" | Measure-Object
Get-ChildItem _cad_eval_pxc -Filter "*.stp" | Measure-Object
```

- [ ] Build `results/pxc_eval_manifest.json` with one row per PXC part:
  `part_nr`, GT JSON path, STEP path, family/type, has_rect_slot, split
  assignment, and QA status.
- [ ] Copy or symlink all newly available, non-empty PXC STEP files into
  `_cad_eval_pxc/` so `cad_eval.py`, `cad_glb_qa.py`, and `cad_ml_fuse.py` can
  score the same set.
- [ ] If a WSCAD download opens as an empty/blank part, do not add it to
  `_cad_eval_pxc/`. Mark it as `wscad_empty` in the manifest and look for an
  official vendor/original CAD source instead.
- [ ] Generate CAD-direct predictions for the expanded PXC set:

```powershell
.venv\Scripts\python.exe step_openings.py _cad_eval_pxc `
  --auto `
  --dir-consensus `
  --check-dirs `
  --out preds_pxc_expanded_cad.json
```

- [ ] Run CAD-only eval on the expanded PXC set and save the baseline:

```powershell
.venv\Scripts\python.exe cad_eval.py `
  --preds preds_pxc_expanded_cad.json `
  --gt-dir "C:\Users\DE00024082\Desktop\JSON" `
  --step-dir _cad_eval_pxc `
  --out results\pxc_cad_eval_expanded.json
```

- [ ] Generate visual QA for the expanded PXC set:

```powershell
.venv\Scripts\python.exe cad_glb_qa.py `
  --preds preds_pxc_expanded_cad.json `
  --gt-dir "C:\Users\DE00024082\Desktop\JSON" `
  --step-dir _cad_eval_pxc `
  --out-dir glb_cad_qa
```

- [ ] Expand the CAD/PXC eval from 3 parts to the full available PXC set only
  after valid STEP geometry exists. If the catalog STEP is empty, that part is
  blocked for CAD eval and cannot help the 90% evidence yet.
- [ ] Split reporting by family in every run: `wscaduniverse`, `PXC`, and any
  future ABB/non-terminal families. General 90% must not hide a zero-PXC model.
- [ ] Create a "do-not-touch" final test list with at least:
  - 20+ wscaduniverse parts,
  - 10+ PXC/cage-clamp/rect-slot parts,
  - several zero-CP or clutter-heavy negatives.

## 3. Data work that can actually move recall

- [ ] Collect or generate labels for at least 300 train parts in the target scope.
  Current notes estimate roughly 111 train parts -> 0.644; 90% likely needs a
  much larger learning curve unless the task is narrowed.
- [ ] Add 10-15 rectangular/push-in slot parts with STEP + human GT. This is the
  specific blind spot blocking CAD+ML fusion and PXC recall.
- [ ] Create `rect_slot_verified/` with the 10-15 verified rect-slot JSON labels.
  Keep these separate from auto-generated labels until QA is complete.
- [ ] For every rect-slot part, save a QA note under `results/rect_slot_qa/` with:
  expected CP count, whether CAD cylinder path sees it, whether `--rect-slots`
  helps, and whether hp_v22 predicts a near-miss within 7mm.
- [ ] Add hard negative PXC parts that look slot-like but are not wire entries.
  The balanced run needs negatives too, otherwise low thresholds will leak FPs.
- [ ] Regenerate labels into a new folder only; never overwrite the active corpus:

```powershell
.venv\Scripts\python.exe step_openings.py all_wscad_stp `
  --label-corpus wscad_corpus_v3 `
  --auto `
  --deflection 0.5
```

- [ ] For PXC/rect-slot parts, do not trust auto labels blindly. Visual QA with
  `cad_glb_qa.py` and compare red CAD detections vs green human GT.
- [ ] Add hard negatives from CAD clutter: screw holes, cosmetic fillets, latch
  pockets, rail edges, vent slots. False positives here are what a low threshold
  will leak.

## 4. CAD-direct path to near-90 for STEP-available parts

- [ ] Keep cylindrical/slot openings on the proven `step_openings.py` path.
- [ ] Re-test `--rect-slots` only after the rect-slot eval set has 10+ examples;
  one example is not enough to tune it.
- [ ] Build a rect-slot tuning sheet with per-part TP/FP/FN, not only pooled F1.
- [ ] Acceptance for CAD-direct:
  - cylindrical/oval slots: >= 95% precision and recall,
  - rect slots: >= 80% F1 before enabling by default,
  - overall PXC eval: >= 85% before claiming progress toward 90.

## 5. ML training experiments after data expansion

- [ ] Start from `run_hp_v22.yaml`, not `run_hp_v23.yaml`; `w_heat=2.0` did not
  improve the current run.
- [ ] Add a data-balanced sampler or loss weighting so PXC/rect-slot parts are
  oversampled until their per-family recall is no longer near zero.
- [ ] Implement an explicit training knob in `train_cp.py`, for example
  `--family-sample-weights PXC=3,wscaduniverse=1` or
  `--oversample-prefix PXC:3`, and record the exact weighting in checkpoint
  metadata.
- [ ] Create `run_hp_v24_balanced_pxc.yaml` from `run_hp_v22.yaml` with:
  `run_name: cp_hp_v24_balanced_pxc`, same `split_seed: 0`, same decode settings,
  plus the new PXC/rect-slot balancing knob.
- [ ] Include the verified rect-slot folder as an extra source for hp_v24:
  `extra_sources: [wscad_corpus_v2, rect_slot_verified]`.
- [ ] Run hp_v24 only after the expanded PXC eval and rect-slot labels exist:

```powershell
.venv\Scripts\python.exe train_cp.py "C:\Users\DE00024082\Desktop\JSON" `
  --config run_hp_v24_balanced_pxc.yaml
```

- [ ] After hp_v24, evaluate with low-threshold no-TTA sweep and per-family
  thresholds. Compare against hp_v22 tuned baseline, not just raw epoch metrics.
- [ ] Add a localization-focused fine-tune stage for rect-slot near-misses:
  lower LR, keep `w_off=5.0`, evaluate 5mm and 7mm match radii separately to
  distinguish classification miss vs 5-7mm localization miss.
- [ ] Try per-family decode thresholds with `tune_thresholds.py`; write thresholds
  only to a copied checkpoint:

```powershell
Copy-Item checkpoints\cp_hp_v22_best.ckpt checkpoints\cp_hp_v22_best_thr.ckpt
.venv\Scripts\python.exe tune_thresholds.py checkpoints\cp_hp_v22_best_thr.ckpt `
  "C:\Users\DE00024082\Desktop\JSON" `
  --extra-source wscad_corpus_v2 `
  --keep-prefixes wscaduniverse,PXC `
  --split-group geometry `
  --thresholds "0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35,0.40" `
  --device cuda `
  --write
```

- [ ] Train at least these controlled variants once the new data exists:
  - `hp_v24_balanced_pxc`: same as v22 + PXC/rect-slot oversampling.
  - `hp_v25_rect_finetune`: init from best balanced model + localization-focused
    fine-tune.
  - `hp_v26_more_context`: keep hierpoint, test larger context only if GPU memory
    allows; otherwise do not spend time here.
  - `hp_v27_ensemble3`: decode-level ensemble of the three best seeds, accepted
    only if it lifts test F1 without FP leakage.

## 6. Diagnostics for every candidate

- [ ] Run `diag_cp_errors.py` at the selected threshold and save JSON.
- [ ] Count false negatives by cause:
  - model blind (`max_heat_near < 0.20`),
  - threshold issue,
  - localization miss just outside 5mm,
  - missing/wrong label.
- [ ] Run `diag_fp_analysis.py` and tag FP families: CAD clutter, rail, screw,
  pocket, duplicate, or label ambiguity.
- [ ] Stop a run family if the same FN class remains dominant after two attempts;
  that means the next lever is data/labels, not hyperparameters.

## 7. Promotion criteria for "90%"

- [ ] Val micro-F1 >= 0.90 on the locked validation split.
- [ ] Test micro-F1 >= 0.85 on the untouched test split.
- [ ] Per-family F1 >= 0.80 for both `wscaduniverse` and `PXC`.
- [ ] Best-vs-last gap <= 0.03, or export only the best checkpoint with explicit
  decode metadata.
- [ ] CAD+ML fusion must be no worse than CAD-only on every eval family.
- [ ] Save the final command, checkpoint, decode threshold(s), and per-family
  confusion counts in `RESULTS.md`.

## 8. Practical next move

- [ ] Wait for `cp_hp_v22_seed1`/seed2 to finish.
- [ ] Run low-threshold no-TTA sweeps for all seeds.
- [ ] Audit/acquire missing PXC STEP files.
- [ ] Build the rect-slot eval/training subset.
- [ ] Only then start `hp_v24_balanced_pxc`.
