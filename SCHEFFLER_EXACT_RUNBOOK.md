# Scheffler-Bründl exact WSCAD runbook

This is the leakage-safe route for joining the public human-labelled mesh data
to byte-identical local WSCAD STEP geometry.

## Frozen data contract

- 71 published train parts
- 20 published validation parts
- 11 published three-rater test parts, locked until final evaluation
- 102 exact original STEP SHA-256 matches in total
- 3 same-catalog but non-exact STEP files excluded

The 91-part development pool is used in two stages. Hyperparameters and the
epoch are selected on 71/20. After that decision is frozen, a new model is
trained from scratch for the fixed number of epochs on all 91 parts. The 11
test parts are never used for training, preprocessing choices, thresholds, or
model selection.

## 1. Recreate and verify the corpus

```powershell
.\.venv\Scripts\python.exe .\prepare_scheffler_wscad_exact.py --dry-run
.\.venv\Scripts\python.exe .\prepare_scheffler_wscad_exact.py
```

The script pins the publisher metadata commit and Dataverse version, verifies
download MD5 values, verifies original STEP SHA-256 values and signatures,
checks OBJ vertex/label lengths, and independently recomputes the three-rater
majority labels.

## 2. Select the semantic model on 71/20

```powershell
.\.venv\Scripts\python.exe .\train_scheffler_semantic.py --preflight-only
.\.venv\Scripts\python.exe .\train_scheffler_semantic.py
```

The default network is the thesis DiffusionNet configuration already encoded
in `diffusionnet.py`. The training receipt records corpus, split, checkpoint,
and code hashes. Operator failures are fatal; a part cannot be silently skipped.

If another GPU training run is active, queue both selection and the fixed
91-part refit without interrupting it:

```powershell
.\queue_scheffler_semantic.ps1
```

## 3. Fixed final refit on all 91 development parts

```powershell
.\.venv\Scripts\python.exe .\refit_scheffler_semantic_91.py --preflight-only
.\.venv\Scripts\python.exe .\refit_scheffler_semantic_91.py
```

This command accepts no hyperparameter or epoch overrides. It takes the best
zero-based validation epoch from the frozen 71/20 receipt, converts it to a
fixed epoch count, combines train+validation, starts from new random weights,
and saves the last epoch. It computes no validation or test score.

## 4. Create reviewable CP candidates from development labels

```powershell
.\.venv\Scripts\python.exe .\derive_scheffler_cp_candidates.py
```

The semantic labels are not human-clicked CP coordinates. The bridge therefore:

- treats `CableEntry` connected components as primary CP candidates;
- treats `Contacting` components as auxiliary evidence, never as a CP count;
- derives point/direction in the labelled OBJ frame;
- maps them to the exact STEP frame only after surface alignment QA;
- uses a STEP free-space probe to check direction sign;
- marks every result `human-region-derived` and `review_required`.

Only 57/71 train, 16/20 validation, and 9/11 test parts contain a CableEntry
region. The remaining 14/4/2 parts are **not zero-CP negatives**. They require
connection-method-aware or manual annotation.

## 5. Open the locked semantic benchmark before

First verify the final 91-refit receipt without reading a test label:

```powershell
.\.venv\Scripts\python.exe .\eval_scheffler_semantic.py --preflight-only
```

Only after the model, preprocessing, and reporting protocol are frozen:

```powershell
.\.venv\Scripts\python.exe .\eval_scheffler_semantic.py `
  --confirm-final-test OPEN_LOCKED_11_ONCE
```

The evaluator creates an irreversible opened marker before loading the labels.
It reports pooled-vertex and macro-per-part accuracy, Dice/F1, IoU/Jaccard,
per-class Contact/CableEntry metrics, and prediction agreement against each of
the three raters. These are semantic segmentation metrics, not CP localization
F1. A CP benchmark needs human approval of the derived points/directions first;
without resolving the two CableEntry-absent test parts, its honest automatic
coverage is 9/11 (81.8%).

## Sources and terminology

- Dataset: https://doi.org/10.7910/DVN/D3ODGT
- Data descriptor: https://doi.org/10.1038/s41597-024-03155-w
- Publisher analysis/metadata: https://github.com/bensch98/eec-analysis

Use `human semantic GT` for the five vertex classes. Use
`human-region-derived CP` for bridge output until a person signs off the final
point and direction. Do not call the latter `human-clicked CP`.
