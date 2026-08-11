# Pretraining TODO

Intent: prepare the pretraining path on the same data scope as v19, but do not
start training until you pass `-Execute` yourself.

Data scope:

- Source: `C:\Users\DE00024082\Desktop\JSON`
- Extra source: `wscad_corpus`
- Keep prefixes: `wscaduniverse,PXC`
- Split: `geometry`, `val_frac=0.20`, `test_frac=0.15`

Current status:

- No `train_cp.py` training process is running.
- No synthetic corpus is used or generated.
- `run_pretrain.yaml` points at the same corpus scope as v19 through
  `extra_sources` and `keep_prefixes`.
- Run name is `cp_knn_tb_pretrain_v1`, so old `cp_knn_pretrain_*` checkpoints are
  not overwritten.
- `run_pretrain_prepare.ps1` is dry-run by default. It only executes when
  `-Execute` is passed.
- The start script refuses to run if another `train_cp.py` is active, if the
  config leaves the v19 terminal-block scope, or if target checkpoints already
  exist without `-Resume`.
- Queue scripts are dry-run by default and need `-Execute` to start training.

## 1. Check the exact pretraining command

Dry-run, prints the command only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pretrain_prepare.ps1 -StartTraining
```

## 2. Start pretraining later

Actually start training on the v19 corpus scope:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pretrain_prepare.ps1 -StartTraining -Execute
```

Resume this same run later:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pretrain_prepare.ps1 -StartTraining -Execute -Resume
```

Logs/artifacts:

- Training log: `train_pretrain.log`
- Checkpoints/history: `checkpoints\cp_knn_tb_pretrain_v1_*`

## 3. Acceptance

- Beat or match v19 best: `val_micro_f1 >= 0.6383`.
- Last eval should stay close to best: gap `<= 0.02`.
- Promote only `checkpoints\cp_knn_tb_pretrain_v1_best.ckpt`, not `_last.ckpt`,
  unless best and last are effectively identical.

Summary command after training finishes:

```powershell
.venv\Scripts\python.exe summarize_cp_runs.py --pattern checkpoints\cp_knn_tb_pretrain_v1_history.json --baseline 0.6383 --max-gap 0.02
```
