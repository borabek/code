#!/usr/bin/env bash
# Dilation A/B: same runs as _human.log but with the annotator's marks grown to the
# corpus convention (_label_targets_dilated, mean 5.48% vs 1.49%). Everything else
# identical so the only variable is label EXTENT.
#   baselines -- human20 0.5858 / human20_pseudo 0.6303 / selftrain_120 0.6764 (CP F1 0.520)
set -u
export PYTHONPATH=_diffusion_net_repo/src   # diffusion_net lives in the vendored repo, not site-packages
PY=.venv/Scripts/python.exe

run () {   # run <name> <ckpt> <extra args...>
  local name=$1 ckpt=$2; shift 2
  echo "--- $name ---"
  $PY train_seg_extra.py --checkpoint-out "$ckpt" --partial-dir _label_targets_dilated \
      --no-extra "$@" 2>&1 | grep -Ev "^ *(epoch|prep)"
  # the pipe hides the exit code -> check the artefact instead, so a crash cannot read as success
  [ -f "$ckpt" ] || { echo "!! $name FAILED: no checkpoint written"; return 1; }
}

echo "=== DILATED LABELS $(date) ==="
run "A2: 71 korpus + 20 genisletilmis insan" results/seg_extra/human20d.pt
run "B2: 71 + 450 pseudo + 20 genisletilmis insan" results/seg_extra/human20d_pseudo.pt \
    --pseudo-dir _pseudo_extra
echo "=== DONE $(date) ==="
