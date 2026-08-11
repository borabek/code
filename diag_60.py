# -*- coding: utf-8 -*-
"""60-PARCA KOK-NEDEN TESHISI: modelin HIC bulamadigi (tp==0) WEI parcalarinin acikligi, modelin GORDUGU
remeshed mesh'te geometrik oyuk olarak VAR mi? Found (tp>0) parcalarla karsilastir.
- zero-recall parcalar found'dan DUZ (dusuk konkavlik) ise -> aciklik mesh'te YOK -> hole-preserving remesh sart
- benzer konkavlik ise -> aciklik VAR ama model firmuyor -> training/model sorunu
CPU-only (GPU yok). geo_cp.concavity + align_frames kullanir."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thesis_remesh, trimesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh
from geo_cp import concavity
from big_arbiter import eligible

# pid -> (jf, stp)
pmap = {p: (jf, stp) for m, p, jf, stp in eligible() if m == "WEI"}
rows = json.load(open("results/big_arbiter_recall_v2_wei.json"))["per_part"]
zero = [r["part_id"] for r in rows if r["tp"] == 0 and r["mfg_cps"] > 0]
found = [r["part_id"] for r in rows if r["tp"] > 0]
print(f"zero-recall: {len(zero)} | found: {len(found)}", flush=True)

import random; random.seed(0)
zsamp = random.sample(zero, min(25, len(zero)))
fsamp = random.sample(found, min(25, len(found)))


def cp_concavity(pids):
    """her parca: uretici CP civarindaki (mesh frame) mesh vertexlerinin max konkavligi."""
    vals = []
    for pid in pids:
        if pid not in pmap: continue
        jf, stp = pmap[pid]
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j["ConnectionPoints"]], float)
            Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j["ConnectionPoints"]], float)
            if not len(G): continue
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, float); F = np.ascontiguousarray(F, np.int64)
            R, t, _ = align_frames(Vr, Vj)
            Gs = (G - t) @ R                                # CP'yi mesh frame'e
            Gds = ((Gd/(np.linalg.norm(Gd,axis=1,keepdims=True)+1e-9)) @ R)
            mesh = trimesh.Trimesh(V, F, process=False)
            sc = concavity(mesh, R=3.0)
            from scipy.spatial import cKDTree
            tree = cKDTree(V)
            part_max = []
            for gi, (g, gd) in enumerate(zip(Gs, Gds)):
                # agiz = seat'ten disariya ~ CP civari; CP'nin R-icindeki vertexlerin max konkavligi
                nb = tree.query_ball_point(g, 6.0)
                part_max.append(float(sc[nb].max()) if nb else 0.0)
            vals.append(np.mean(part_max))
        except Exception:
            continue
    return np.array(vals)


zc = cp_concavity(zsamp)
fc = cp_concavity(fsamp)
print(f"\nZERO-RECALL parcalar (model HIC bulamadi): CP-konkavlik ort {zc.mean():+.3f} med {np.median(zc):+.3f} ({len(zc)} parca)")
print(f"FOUND parcalar (model buldu):             CP-konkavlik ort {fc.mean():+.3f} med {np.median(fc):+.3f} ({len(fc)} parca)")
print(f"\nKARAR:")
if zc.mean() < fc.mean() - 0.05:
    print(f"  zero-recall BELIRGIN DUZ (found'dan {fc.mean()-zc.mean():.3f} dusuk konkav)")
    print(f"  -> ACIKLIK MESH'TE YOK/ZAYIF -> etiket kurtaramaz -> HOLE-PRESERVING REMESH sart")
else:
    print(f"  zero-recall ~ found (konkavlik benzer)")
    print(f"  -> ACIKLIK VAR ama model firmuyor -> TRAINING/MODEL sorunu (etiket yaklasimi/kapasite)")
