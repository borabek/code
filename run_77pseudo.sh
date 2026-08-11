#!/usr/bin/env bash
# +PSEUDO lever on the 77-label winner. C2 warned pseudo inflates val but tanked the arbiter (0.464).
# Adopt ONLY if arbiter beats human77c's 0.667/mean 0.698. seed 0 first for a quick read.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
echo "=== human77c + pseudo seed 0 $(date) ==="
$PY train_seg_extra.py --no-extra --pseudo-dir _pseudo_extra \
    --partial-dir _label_targets _label_targets_2 _label_targets_3 \
    --partial-target connection --seed 0 --checkpoint-out results/seg_extra/human77cp_s0.pt 2>&1 | grep -E "DONE best|self-training|partial-human"
echo "=== DONE $(date) ==="
