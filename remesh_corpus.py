# -*- coding: utf-8 -*-
"""Build the thesis-style uniform-remeshed teacher corpus: every Desktop\\JSON part
isotropic-remeshed to ~6000 uniform vertices (tessellation normalised so JSON-train and
STEP-infer match), CPs kept at their 3D locations. Output = a new corpus dir; read-only
Desktop\\JSON untouched.  Usage: .venv/Scripts/python.exe remesh_corpus.py --out _remeshed
"""
import argparse, os, json, traceback
import numpy as np
import json_dataset as jd
from thesis_remesh import remesh_uniform

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=r"C:\Users\DE00024082\Desktop\JSON")
ap.add_argument("--out", default="_remeshed")
ap.add_argument("--target", type=int, default=6000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

n = ok = fail = 0
for p in jd.iter_parts(a.src):
    n += 1
    pn = str(p.part_nr)
    try:
        V = np.asarray(p.vertices, float); F = np.asarray(p.faces, int)
        Vr, Fr = remesh_uniform(V, F, target=a.target)
        if len(Vr) < 100 or len(Fr) < 100:
            raise ValueError(f"remesh degenerate ({len(Vr)}v/{len(Fr)}f)")
        lo = Vr.min(0); hi = Vr.max(0)
        cps = [{"Index": i, "Name": str(p.cp_names[i]) if i < len(p.cp_names) else str(i),
                "Point": {"X": float(c[0]), "Y": float(c[1]), "Z": float(c[2])},
                "InsertDirection": {"X": float(d[0]), "Y": float(d[1]), "Z": float(d[2])}}
               for i, (c, d) in enumerate(zip(p.cp_points, p.cp_directions))]
        obj = {"PartNr": pn,
               "Graphic3d": {"Points": [{"X": float(v[0]), "Y": float(v[1]), "Z": float(v[2])} for v in Vr],
                             "Indices": Fr.reshape(-1).astype(int).tolist()},
               "BoundingBox": {"Dimension": {"X": float(hi[0]-lo[0]), "Y": float(hi[1]-lo[1]), "Z": float(hi[2]-lo[2])},
                               "Location": {"X": float(lo[0]), "Y": float(lo[1]), "Z": float(lo[2])}},
               "ConnectionPoints": cps}
        json.dump(obj, open(os.path.join(a.out, pn + ".json"), "w"))
        ok += 1
    except Exception as exc:
        fail += 1
        print(f"  FAIL {pn}: {exc}")
    if n % 50 == 0:
        print(f"  {n}/479 ... ok={ok} fail={fail}", flush=True)

print(f"DONE: {ok} ok, {fail} fail -> {a.out}/")
