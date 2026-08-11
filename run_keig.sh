#!/usr/bin/env bash
# k_eig 64 -> 96 at the PRODUCT resolution (6k). Motivation from the 9k result: DiffusionNet works in
# the SPECTRAL domain, so the finest structure it can represent is set by the number of eigenvectors,
# NOT by the vertex count -- adding vertices at fixed k_eig gave recall but cost precision (net zero,
# and the arbiter rejected it). More BASIS is the untested counterpart, and what we miss is small
# openings. 3 seeds, because 9k proved single seeds do NOT predict the ensemble.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 0 1 2; do
  echo "=== k_eig96 seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --k-eig 96 \
      --partial-dir _label_targets _label_targets_2 _label_targets_3 \
      --partial-target connection --seed $s \
      --checkpoint-out results/seg_extra/human77c_keig96_s$s.pt 2>&1 | grep -E "DONE best"
done
echo "=== DONE $(date) ==="
