# Training Plateau TODO

Context from the v19 logs:

- `cp_knn_v19_best.ckpt` is still good: best epoch is 50 with `val_micro_f1=0.6383`.
- `cp_knn_v19_last.ckpt` drifted: epoch 95 is down to `val_micro_f1=0.5431` while train loss kept falling from about `0.699` to `0.438`.
- v19 scope is small and specialist: `wscaduniverse,PXC`, 157 parts total, 109 train, 27 val, 21 test.
- v19 train config in checkpoint: `augment=4`, `lr=0.001`, `centernet_alpha=1.5`, `hard_mining=False`, no jitter/reflect/dropout/cable augmentation.
- The speed work touched spatial patching, graph prebuild, feature prebuild, and graph batching. Treat those as suspect until isolated tests say the math is unchanged.

Current status:

- v19 runaway resume processes were stopped.
- Code guardrails are implemented: scheduler/early-stop metadata, resume warnings, LR logging, `min_delta`, final best-vs-last collapse warning, OOM gradient-discard guard.
- Focused tests pass: `python -m pytest test_core.py tests\test_cp.py tests\test_split.py -q` -> 41 passed.
- v19 diagnostics were generated under `results/`.
- Training is not running now. The accidental v20 queue was stopped during epoch 0.
- Pretraining preparation lives in `PRETRAINING_TODO.md`; it uses the same terminal-block scope as v19, not a synthetic corpus. Queue scripts are dry-run by default and need `-Execute` to start training.

## 0. Immediate safety

- [x] Stop using `checkpoints/cp_knn_v19_last.ckpt` for deployment or CAD fusion.
- [x] Use `checkpoints/cp_knn_v19_best.ckpt` as the recovery point.
- [x] Do not resume v19 further unless the goal is only investigation. Fresh run name for every real experiment.
- [x] Copy the v19 numbers into the experiment log:
  - best: epoch 50, micro F1 `0.6383`, accuracy `49.7%`, F1 `60.9%`
  - last: epoch 95, micro F1 `0.5431`, accuracy `39.7%`, F1 `52.1%`

## 1. Fix checkpoint/run metadata first

- [x] Add these fields to `train_config` in `train_cp.py`: `epochs`, `eval_every`, `patience`, `lr_schedule`, `lr_decay_every`, `lr_decay_rate`, `warmup_epochs`, `ckpt_every`, `snapshot_every`.
- [x] Add at least `lr_schedule`, `lr_decay_every`, `lr_decay_rate`, and `warmup_epochs` to `_RESUME_CRITICAL` in `cp_regressor.py`.
- [x] Print current LR at each validation eval, or at least once per epoch.
- [x] Add a resume warning if the checkpoint metadata is missing scheduler/early-stop fields.
- [x] Add a small test that resume refuses or warns when `lr_schedule` or `patience` changes.

Why: v19 checkpoint metadata records `lr`, but not the schedule/patience/epoch budget. That makes resume experiments too easy to compare incorrectly.

## 2. Make overfit impossible to miss

- [x] For terminal-block specialist runs, use shorter patience: `eval_every=5`, `patience=3` or `4`, not 12.
- [x] Add `min_delta` for best metric, for example only reset stale counter if `micro_f1` improves by `>= 0.005`.
- [x] Keep `--snapshot-every 5` so later collapse cannot hide the useful mid-run checkpoint.
- [x] Make final console summary show best epoch vs last epoch gap.
- [x] Warn in the run summary if `last_micro_f1` is more than `0.03` below best.

## 3. Run the v20 controlled baseline

Goal: same data scope, but with schedule/early-stop guardrails and light regularization restored.

Do this only after the pretraining step if you want scratch-vs-pretrained
comparison. For pretrained fine-tune, use `run_finetune_pretrain_v20.yaml`
instead of the scratch v20 config.

```powershell
python train_cp.py "C:\Users\DE00024082\Desktop\JSON" `
  --backbone knngraph `
  --run-name cp_knn_v20 `
  --extra-source wscad_corpus `
  --keep-prefixes wscaduniverse,PXC `
  --split-group geometry `
  --val-frac 0.20 `
  --test-frac 0.15 `
  --max-gpu-verts 7000 `
  --augment 4 `
  --aug-jitter-frac 0.005 `
  --aug-reflect `
  --aug-dropout-frac 0.10 `
  --aug-cable-frac 0.20 `
  --heat-loss centernet `
  --centernet-alpha 1.5 `
  --w-heat 1.0 `
  --w-off 5.0 `
  --w-dir 2.0 `
  --lr 0.0005 `
  --lr-schedule cosine `
  --weight-decay 0.0005 `
  --grad-clip 1.0 `
  --eval-every 5 `
  --patience 4 `
  --epochs 70 `
  --ckpt-every 5 `
  --snapshot-every 5 `
  --knn-cache-dir knn_cache_real
```

Acceptance:

- [ ] Best micro F1 should reach at least v19 best: `>= 0.6383`.
- [ ] Last checkpoint should stay close to best: gap `<= 0.02`.
- [ ] Recall should not collapse after epoch 50.

## 4. Ablations to identify the actual cause

Run these as separate run names, same split/scope/cache:

- [ ] `cp_knn_v20_no_aug_reg`: same as v20 but no jitter/reflect/dropout/cable. Tests whether v19 collapse was mainly weak augmentation.
- [ ] `cp_knn_v20_lr1e3`: same as v20 but `--lr 0.001`. Tests whether LR was too hot.
- [ ] `cp_knn_v20_plateau`: same as v20 but `--lr-schedule plateau --lr-decay-every 2 --lr-decay-rate 0.5`. Tests whether metric-driven LR fixes the post-50 slide.
- [ ] `cp_knn_v20_hard`: same as v20 plus `--hard-mining`. Only keep it if per-family recall improves without FP explosion.

Keep the one with the best validation curve shape, not just one lucky peak.

## 5. Test the speed refactor

- [x] Unit test: one unbatched graph forward/backward vs `_batch_accumulate` on two small graphs gives matching loss/gradients within tolerance.
- [x] Unit test: a patched part has graph weights that sum to 1.0 across its patches.
- [x] Unit test: hard-mining EMA uses the right unit. Decision: it mines prepared graphs/patches; patch `graph_weight` keeps patched part gradient budget normalized.
- [x] Unit test: `spatial_patches` plus train target slicing keeps every CP peak recoverable.
- [x] Smoke test replacement: direct batched-vs-solo forward/backward unit test now checks the same math without spending epochs.
- [x] Audit OOM fallback: if a GPU batch OOMs mid-accumulation, previous accumulated gradients should not be silently discarded.

## 6. Diagnose best vs last

Run diagnostics against both best and last:

```powershell
python diag_cp_errors.py checkpoints/cp_knn_v19_best.ckpt "C:\Users\DE00024082\Desktop\JSON" `
  --extra-source wscad_corpus --keep-prefixes wscaduniverse,PXC `
  --split-group geometry --thr 0.30 --json results\v19_best_cp_errors.json

python diag_cp_errors.py checkpoints/cp_knn_v19_last.ckpt "C:\Users\DE00024082\Desktop\JSON" `
  --extra-source wscad_corpus --keep-prefixes wscaduniverse,PXC `
  --split-group geometry --thr 0.30 --json results\v19_last_cp_errors.json

python diag_fp_analysis.py checkpoints/cp_knn_v19_best.ckpt "C:\Users\DE00024082\Desktop\JSON" `
  --extra-source wscad_corpus --keep-prefixes wscaduniverse,PXC `
  --split-group geometry --thr 0.30

python tune_thresholds.py checkpoints/cp_knn_v19_best.ckpt "C:\Users\DE00024082\Desktop\JSON" `
  --extra-source wscad_corpus --keep-prefixes wscaduniverse,PXC `
  --split-group geometry
```

Look for:

- [x] Last model losing recall on the same families/parts.
- [x] FP increase around CAD clutter vs near-miss localization errors.
- [x] Threshold sensitivity: if `thr=0.30` is not optimal anymore, write decode thresholds only after comparing best/test.

## 7. CAD fusion check

- [x] Re-run the existing CAD+ML fusion with v19 best, not v17 and not v19 last:

```powershell
python cad_ml_fuse.py --cad-preds preds_cad_pxc_best.json `
  --gt-dir "C:\Users\DE00024082\Desktop\JSON" `
  --step-dir _cad_eval_pxc `
  --ckpt checkpoints/cp_knn_v19_best.ckpt `
  --device cpu
```

- [x] If fusion is worse than CAD-only, inspect which ML predictions are subtracting value before tuning training again.

## 8. Data/label cleanup after v19

- [ ] Regenerate WSCAD labels into a new folder only, never overwrite the active `wscad_corpus` in place:

```powershell
python step_openings.py all_wscad_stp --label-corpus wscad_corpus_v2 --auto
```

- [ ] Train a fresh v21 against `wscad_corpus_v2` and compare against v19/v20.
- [ ] Because v19 val has only 27 parts, repeat the final candidate over 3 split seeds before trusting tiny metric differences.

## 9. Stop criteria

- [ ] Keep a run only if it beats or matches v19 best and does not collapse later.
- [ ] Prefer a stable curve over a single high spike.
- [ ] Promote only a `best.ckpt`, never a `last.ckpt`, unless best and last are effectively identical.
