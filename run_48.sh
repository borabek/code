#!/usr/bin/env bash
# The real test: 71 corpus + 48 human labels (20 batch-1 + 28 batch-2), connection-channel masked
# loss. Baselines -- 71-only 0.622 val Conn IoU; product selftrain_120 arbiter CP F1 0.520.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
echo "=== 48 HUMAN (batch1+batch2) $(date) ==="
$PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 \
    --partial-target connection --checkpoint-out results/seg_extra/human48c.pt 2>&1 | grep -Ev "^ep"
echo "=== DONE $(date) ==="
