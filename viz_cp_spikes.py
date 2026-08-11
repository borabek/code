# -*- coding: utf-8 -*-
"""Bulletproof CP markers: a big MAGENTA SPIKE (cylinder+cone, arrow) sticking OUT of every
CP along its approach direction. Even if a viewer ignores vertex colours, a spike protruding
from the mesh is an unmistakable geometric bump -- unlike a flush cube/sphere. Grey mesh +
spikes, straight from the frozen candidate file (no model needed). One GLB per CP-part.

Usage: .venv/Scripts/python.exe viz_cp_spikes.py --out _scheffler_cp_glb
"""
import argparse, json, os
import numpy as np
import trimesh

GREY = (165, 165, 165, 255); MAG = (235, 30, 220, 255)


def load_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, int)


def spike(point, direction, r, h):
    d = np.asarray(direction, float); n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    shaft = trimesh.creation.cylinder(radius=r, height=h, sections=16)
    tip = trimesh.creation.cone(radius=r * 2.0, height=h * 0.5, sections=16)
    tip.apply_translation([0, 0, h / 2 + h * 0.25])
    arrow = trimesh.util.concatenate([shaft, tip])
    T = trimesh.geometry.align_vectors([0, 0, 1], d)
    arrow.apply_transform(T)
    arrow.apply_translation(point + d * (h * 0.35))          # base near surface, points OUT
    arrow.visual.vertex_colors = np.tile(MAG, (len(arrow.vertices), 1)).astype(np.uint8)
    return arrow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="results/scheffler_cp_bridge/development_candidates.json")
    ap.add_argument("--out", default="_scheffler_cp_glb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = json.load(open(a.cands, encoding="utf-8"))
    n = 0
    for p in d["parts"]:
        cs = p.get("candidates") or []
        if not cs:
            continue
        V, F = load_obj(p["source_obj"])
        diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
        r = diag * 0.012; h = diag * 0.16                    # long, obvious spikes
        mesh = trimesh.Trimesh(vertices=V, faces=F,
                               vertex_colors=np.tile(GREY, (len(V), 1)).astype(np.uint8), process=False)
        parts = [mesh]
        for c in cs:
            pt = np.asarray(c["obj_frame"]["entry_point"], float)
            dv = np.asarray(c["obj_frame"]["approach_vector_after_step_probe"], float)
            parts.append(spike(pt, dv, r, h))
        trimesh.util.concatenate(parts).export(os.path.join(a.out, f"{p['part_id']}_CP.glb"))
        n += 1
        if n % 20 == 0:
            print(f"  {n} parts...", flush=True)
    print(f"wrote {n} CP-spike GLBs -> {a.out}/<id>_CP.glb  (MAGENTA spikes = CP points, pointing out)")
    print("DONE")


if __name__ == "__main__":
    main()
