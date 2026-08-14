#!/bin/bash
# R3 plato testi + robot uctan-uca. Ikisi de _hw_r3.txt (145 leakage-siz WEI part) uzerinde.
# BA_ALLOW_SEEN=1: --only-parts acik liste zaten leakage korumasi (liste egitilen part icermiyor).
set -e
cd /c/Users/DE00024082/Desktop/code
PY=".venv/Scripts/python.exe"
export PYTHONPATH=_diffusion_net_repo/src
HELD=$(cat _hw_r3.txt)

echo "############ 1) ONCEKI URUN (recall_hard_s2) — WEI held-out ############"
BA_ALLOW_SEEN=1 $PY big_arbiter.py --ckpts results/seg_extra/recall_hard_s2.pt --only-mfg WEI \
  --only-parts $HELD --axis-aware --cluster-mm 5 --min-v 30 --vertex-conf 0.5 --tag prev_wei 2>&1 | tail -20

echo "############ 2) R3 URUN (recall_r3_s2) — AYNI WEI held-out ############"
BA_ALLOW_SEEN=1 $PY big_arbiter.py --ckpts results/seg_extra/recall_r3_s2.pt --only-mfg WEI \
  --only-parts $HELD --axis-aware --cluster-mm 5 --min-v 30 --vertex-conf 0.5 --tag r3_wei 2>&1 | tail -20

echo "############ 3) SET AUDIT (birebir ayni part seti mi) ############"
$PY audit_sets.py prev_wei r3_wei 2>&1 | tail -15
