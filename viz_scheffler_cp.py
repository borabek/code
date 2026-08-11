# -*- coding: utf-8 -*-
"""Review visualiser for the human-region-derived CP candidates. One OBJ (+ .mtl) per dev
part with candidates: the Scheffler mesh (grey) + a GREEN cube at each CableEntry-derived
entry point + a RED wedge along its approach vector. Uses a .mtl material file so Windows
3D Viewer renders the CP markers in ACTUAL colour (without it every object is uniform grey
and the small markers vanish into the mesh). Open the .obj in Windows 3D Viewer.

Usage: .venv/Scripts/python.exe viz_scheffler_cp.py --out _scheffler_cp_review
"""
import argparse, json, os
import numpy as np

_CUBE = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
_CT = [(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
       (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
_MTL = ("newmtl mesh\nKd 0.72 0.72 0.72\n\n"
        "newmtl cp_green\nKd 0.0 0.9 0.1\n\n"
        "newmtl dir_red\nKd 0.95 0.1 0.1\n")


def load_obj(path):
    V, F = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, int)


def cubes(pts, r, base):
    vl, fl, b = [], [], base
    for p in pts:
        for c in _CUBE:
            vl.append(f"v {p[0]+c[0]*r:.3f} {p[1]+c[1]*r:.3f} {p[2]+c[2]*r:.3f}")
        for t in _CT:
            fl.append(f"f {b+t[0]+1} {b+t[1]+1} {b+t[2]+1}")
        b += 8
    return vl, fl, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="results/scheffler_cp_bridge/development_candidates.json")
    ap.add_argument("--out", default="_scheffler_cp_review")
    ap.add_argument("--frame", default="obj_frame")
    a = ap.parse_args()
    d = json.load(open(a.cands, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "cp.mtl"), "w").write(_MTL)
    n_viz = 0; missing = []
    for p in d["parts"]:
        cs = p.get("candidates") or []
        if not cs:
            missing.append(p["part_id"]); continue
        V, F = load_obj(p["source_obj"])
        pts = np.array([c[a.frame]["entry_point"] for c in cs], float)
        dirs = np.array([c[a.frame]["approach_vector_after_step_probe"] for c in cs], float)
        diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
        r = max(diag * 0.03, 1.5); L = diag * 0.14        # bigger, clearly visible markers
        lines = [f"# {p['part_id']} [{p['split']}]: {len(cs)} CP-candidate (review_required)",
                 "mtllib cp.mtl", "o mesh", "usemtl mesh"]
        for v in V:
            lines.append(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}")
        off = len(V)
        vl, cp_fl, off = cubes(pts, r, off)                # green cube verts
        dvl, dir_fl, b = [], [], off                        # red direction wedges
        for pt, dv in zip(pts, dirs):
            end = pt + dv * L
            dvl += [f"v {pt[0]:.3f} {pt[1]:.3f} {pt[2]:.3f}",
                    f"v {end[0]:.3f} {end[1]:.3f} {end[2]:.3f}",
                    f"v {pt[0]+r*0.4:.3f} {pt[1]+r*0.4:.3f} {pt[2]:.3f}"]
            dir_fl.append(f"f {b+1} {b+2} {b+3}"); b += 3
        lines += vl + dvl
        lines.append("usemtl mesh")
        for t in F:
            lines.append(f"f {t[0]+1} {t[1]+1} {t[2]+1}")
        lines.append("o CP_candidate_green"); lines.append("usemtl cp_green"); lines += cp_fl
        lines.append("o CP_direction_red"); lines.append("usemtl dir_red"); lines += dir_fl
        open(os.path.join(a.out, f"{p['part_id']}.obj"), "w").write("\n".join(lines))
        n_viz += 1
    open(os.path.join(a.out, "_MISSING_CABLEENTRY_manual_review.txt"), "w").write("\n".join(missing))
    print(f"wrote {n_viz} coloured review OBJs (+cp.mtl) -> {a.out}/  GREEN=CP candidate, RED=direction")
    print(f"{len(missing)} parts have NO CableEntry region (manual): _MISSING_CABLEENTRY_manual_review.txt")
    print("DONE")


if __name__ == "__main__":
    main()
