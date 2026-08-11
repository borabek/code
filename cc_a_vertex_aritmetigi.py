# -*- coding: utf-8 -*-
"""CC-A: her GT acikligina 6000-vertex mesh'te KAC VERTEX dusuyor? (rejim ayrimli)

SORU: cok-CP parcalarda aday uretimi neden yetmiyor -- ag mi kor, yoksa esik mi IMKANSIZ?

BU OLCUM MODEL KULLANMAZ. Sadece geometri: GT acikliginin cevresinde kac mesh vertex'i VAR?
Bu sayi min_v esiginin altindaysa, segmentasyon MUKEMMEL olsa bile aday olusamaz -- cunku
cp_openings bir bileseni ancak >= min_v vertex iceriyorsa aday sayar. Yani sorun ogrenme degil
ARITMETIK olur ve cozumu yeniden egitim degil, cozunurluk/esik olur.

(Modelsiz olmasi ayrica onemli: GPU'yu mesgul eden cikarimla cakismaz.)
"""
import os, json, sys
import collections
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trimesh
import thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh
from big_arbiter import eligible
from cp_geometry import mouths_for

MIN_V_OLD, MIN_V_NEW = 30, 10          # cp_config: eski/yeni post-isleme esikleri
RADII = (2.0, 3.0, 4.0)                # aciklik yaricapi mertebesinde pencereler
HIGH_CP = 8                            # rejim siniri (olculen medyanlar: dusuk 2, yuksek 12)


def analyse(pid, mfg, jf, stp):
    j = json.load(open(jf, encoding="utf-8-sig"))
    G = np.array([[c["Point"][k] for k in "XYZ"] for c in j["ConnectionPoints"]], float)
    if not len(G):
        return None
    Gd = np.array([[c["InsertDirection"][k] for k in "XYZ"] for c in j["ConnectionPoints"]], float)
    Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
    Vj = np.array([[q[k] for k in "XYZ"] for q in j["Graphic3d"]["Points"]], float)

    Vr, Fr = step_to_mesh(stp)
    V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
    V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
    R, t, _ = align_frames(Vr, Vj)
    Gm = (G - t) @ R; Gdm = Gd @ R
    body = trimesh.Trimesh(V, F, process=False)
    mouths, _ = mouths_for(body, Gm, Gdm)

    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    counts = {r: np.array([len(tree.query_ball_point(m, r)) for m in mouths]) for r in RADII}
    d, _ = cKDTree(V).query(V, k=2)
    spacing = float(np.median(d[:, 1]))
    area = float(body.area)
    return {"pid": pid, "mfg": mfg, "n_gt": len(Gm), "n_vert": len(V),
            "spacing_mm": spacing, "area_mm2": area,
            "vert_per_mm2": len(V) / max(area, 1e-9),
            "counts": {r: counts[r].tolist() for r in RADII}}


def main(limit_per_regime=45):
    parts = list(eligible())
    # rejim ayrimi icin once GT sayilarina bak (ucuz: sadece JSON okur)
    lo, hi = [], []
    for mfg, pid, jf, stp in parts:
        try:
            n = len(json.load(open(jf, encoding="utf-8-sig"))["ConnectionPoints"])
        except Exception:
            continue
        (hi if n >= HIGH_CP else lo).append((mfg, pid, jf, stp, n))
    rng = np.random.RandomState(0)
    sel_lo = [lo[i] for i in rng.choice(len(lo), min(limit_per_regime, len(lo)), replace=False)]
    sel_hi = [hi[i] for i in rng.choice(len(hi), min(limit_per_regime, len(hi)), replace=False)]
    print(f"havuz: dusuk-CP {len(lo)} parca, cok-CP {len(hi)} parca "
          f"-> orneklem {len(sel_lo)} + {len(sel_hi)}\n", flush=True)

    out = {}
    for tag, sel in [("DUSUK-CP", sel_lo), ("COK-CP", sel_hi)]:
        rows = []
        for mfg, pid, jf, stp, n in sel:
            try:
                r = analyse(pid, mfg, jf, stp)
            except Exception as e:
                print(f"  {pid}: atlandi ({type(e).__name__})", flush=True); continue
            if r: rows.append(r)
        if not rows:
            continue
        allc = {r_: np.concatenate([np.array(x["counts"][r_]) for x in rows]) for r_ in RADII}
        print(f"=== {tag} ({len(rows)} parca, {sum(x['n_gt'] for x in rows)} GT aciklik) ===")
        print(f"  vertex araligi medyan {np.median([x['spacing_mm'] for x in rows]):.2f} mm | "
              f"vertex yogunlugu {np.median([x['vert_per_mm2'] for x in rows]):.3f} /mm2 | "
              f"GT/parca medyan {np.median([x['n_gt'] for x in rows]):.0f}")
        for r_ in RADII:
            c = allc[r_]
            print(f"  r={r_}mm  GT basi vertex: medyan {np.median(c):5.1f}  %25 {np.percentile(c,25):5.1f}  "
                  f"| min_v30 ALTINDA %{100*(c<MIN_V_OLD).mean():4.1f}  | min_v10 altinda %{100*(c<MIN_V_NEW).mean():4.1f}")
        out[tag] = {"n_parts": len(rows), "n_gt": int(sum(x["n_gt"] for x in rows)),
                    "spacing_mm": float(np.median([x["spacing_mm"] for x in rows])),
                    "vert_per_mm2": float(np.median([x["vert_per_mm2"] for x in rows])),
                    "area_mm2": float(np.median([x["area_mm2"] for x in rows])),
                    "gt_per_part": float(np.median([x["n_gt"] for x in rows])),
                    "below_min_v30": {str(r_): float((allc[r_] < MIN_V_OLD).mean()) for r_ in RADII},
                    "below_min_v10": {str(r_): float((allc[r_] < MIN_V_NEW).mean()) for r_ in RADII},
                    "median_verts": {str(r_): float(np.median(allc[r_])) for r_ in RADII}}
        print()
    json.dump(out, open("results/cc_a_vertex_aritmetigi.json", "w"), indent=1)
    print("makbuz -> results/cc_a_vertex_aritmetigi.json")


if __name__ == "__main__":
    main()
