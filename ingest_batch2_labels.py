# -*- coding: utf-8 -*-
"""Ingest the user's batch-2 label files from Downloads into _label_targets_2, with STRICT validation.

A label file is only accepted if its line count EXACTLY matches the part's OBJ vertex count -- a
mismatch means the labels are misaligned (garbage) and must be rejected, not silently used.
Reports per part: labelled / skipped / invalid, and the CableEntry vertex count.
"""
import os, glob, shutil
import numpy as np
from region_label_helper import load_obj

import sys
DL = "C:/Users/DE00024082/Downloads"
ROOT = sys.argv[1] if len(sys.argv) > 1 else "_label_targets_2"
CE = 3

pids = sorted(os.path.basename(os.path.normpath(d)) for d in glob.glob(f"{ROOT}/*/") if os.path.isdir(d))
labelled, skipped, invalid = [], [], []
for pid in pids:
    of = f"{ROOT}/{pid}/{pid}.obj"
    lf = f"{DL}/{pid}.labels.txt"
    if not os.path.exists(lf):
        skipped.append(pid); continue
    V, F = load_obj(of)
    try:
        L = np.array([int(x) for x in open(lf).read().split()], np.int64)
    except Exception as e:
        invalid.append((pid, f"parse error {e}")); continue
    if len(L) != len(V):
        invalid.append((pid, f"vertex MISMATCH: {len(L)} labels vs {len(V)} verts")); continue
    if set(np.unique(L)) - set(range(5)):
        invalid.append((pid, f"bad classes {set(np.unique(L))}")); continue
    nce = int((L == CE).sum())
    dst = f"{ROOT}/{pid}/{pid}.labels.txt"
    shutil.copy(lf, dst)
    labelled.append((pid, len(V), nce, 100 * nce / len(V)))

print(f"=== INGEST: {len(labelled)} labelled, {len(skipped)} skipped, {len(invalid)} invalid ===\n")
print(f"{'part':10s} {'verts':>6s} {'CE_vtx':>7s} {'CE%':>6s}")
for pid, nv, nce, pct in labelled:
    flag = "  <-- 0 CableEntry!" if nce == 0 else ""
    print(f"{pid:10s} {nv:>6} {nce:>7} {pct:5.1f}%{flag}")
if invalid:
    print("\nINVALID (NOT ingested):")
    for pid, why in invalid: print(f"  {pid}: {why}")
print(f"\nSKIPPED (no label file -- user's screw-only/flat-plate parts):")
print("  " + " ".join(skipped))
