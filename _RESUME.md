# RESUME (paused 2026-07-20, PC shutdown ~1.5 h)

## Restart command
```bash
bash run_dilated.sh > _dilated.log 2>&1      # A2 is DONE, only B2 needs to rerun
```
`run_dilated.sh` reruns both arms. A2 already produced `results/seg_extra/human20d.pt`
(val Connection IoU **0.5837**) — to save ~15 min, comment out the A2 `run` line and
keep only B2. **PYTHONPATH=_diffusion_net_repo/src is required** (it is set inside the
script; a missing one is what made the first attempt crash while reporting exit 0).

## Where we are
The 20 human CableEntry labels HURT, and the convention hypothesis is now DISPROVEN.

| run | val Connection IoU | arbiter CP F1 |
|---|---|---|
| **selftrain_120 (PRODUCT, unchanged)** | **0.676** | **0.520** |
| 71-only baseline | 0.622 | — |
| B  = 71 + 450 pseudo + 20 human (tight) | 0.630 | 0.351 |
| A  = 71 + 20 human (tight, 1.49 %) | 0.5858 | 0.308 |
| **A2 = 71 + 20 human (DILATED, 5.48 %)** | **0.5837** | not measured |
| B2 = 71 + 450 pseudo + 20 dilated | INTERRUPTED — rerun | — |

**A2 ≈ A** ⇒ label EXTENT was not the cause. Dilation worked as intended
(`dilate_partial_labels.py`, 1.49 % → 5.48 %, visually still on the openings, no housing
spill — `results/_dilated_check.png`) but bought nothing.

## Next hypothesis (ready to run, no extra human work)
OBJECTIVE MISMATCH, not extent: the corpus parts train a 5-class loss, the 20 human parts
train a masked BCE over 1 of 5 classes → the model is pulled toward two different objectives.

`make_hybrid_labels.py` (written, NOT yet run) builds full 5-class labels: the product
model supplies Housing/Contact/SnapPoint/LabelSurface, the human marks overwrite the
CableEntry channel. Then train with the SAME loss as the corpus (`--pseudo-dir
_label_targets_hybrid`, **not** `--partial-dir`). Thesis-endorsed (conclusion line 2157
asks for exactly this prediction→labelling-tool feedback loop).

```bash
export PYTHONPATH=_diffusion_net_repo/src
.venv/Scripts/python.exe make_hybrid_labels.py            # -> _label_targets_hybrid/
.venv/Scripts/python.exe train_seg_extra.py --no-extra \
    --checkpoint-out results/seg_extra/human20h.pt --pseudo-dir _label_targets_hybrid
```

## Decision rule (binding, set before seeing results)
Adopt a new product ONLY if it beats the arbiter **CP F1 0.520**
(`measure_cp_defs.py --ckpt <ckpt>`). Val Connection IoU alone is NOT sufficient — it is
scored against corpus-convention val labels and is biased toward the corpus. If nothing
beats 0.520, record a measured negative; do not report it as "almost there".

## Housekeeping
- No python processes left running; GPU released.
- locked-11 test set still UNTOUCHED.
- Product model unchanged: `results/seg_extra/selftrain_120.pt`.
- Full write-up already in `PRODUCT_MODEL.md` (§ "First 20 human labels ARRIVED").
