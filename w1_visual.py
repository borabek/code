# -*- coding: utf-8 -*-
"""W1 GORSEL ADJUDICATION: WEI parca -> mesh (gri) + manufacturer CP (YESIL) + model tahmin (KIRMIZI).
Kirmizi-only (FP) = modelin bulup manufacturer'in listelemedigi aciklik. Kullanici bakip
'gercek kablo girisi mi?' der -> GT eksik mi (0.58 pesimist mi) belli olur. GLB -> Windows 3D Viewer."""
import os, sys, json
import numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP

dev = "cuda"
held = set(open("_hw_r3.txt").read().split()[:12])
parts = [(m, p, jf, s) for m, p, jf, s in eligible() if m == "WEI" and p in held][:3]
model, meta = load_any("results/seg_extra/recall_hard_s2.pt", dev=dev)[:2]
os.makedirs("results/w1_glb", exist_ok=True)


def sphere(c, r, color):
    s = trimesh.creation.uv_sphere(radius=r); s.apply_translation(c)
    s.visual.vertex_colors = np.tile(color, (len(s.vertices), 1)); return s


for m, pid, jf, stp in parts:
    j = json.load(open(jf, encoding="utf-8-sig"))
    Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
    G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j["ConnectionPoints"]], float)
    Vr, Fr = step_to_mesh(stp); V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
    V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
    _, pb = D.predict(model, meta, V, F, device=dev, op_cache_dir=OP, return_probs=True); pb = np.asarray(pb, float)
    cps = cp_openings.connection_points(V, F, pb.argmax(-1), min_v=30, classes=(CE, CT), dedupe_mm=10.0,
                                        probs=pb, vertex_conf=0.5, ct_depth_min_mm=1.0, cluster_mm=5.0)
    R, t, _ = align_frames(Vr, Vj)
    Gm = (G - t) @ R                                    # manufacturer CP -> mesh frame
    P = np.array([np.asarray(c["point"]) for c in cps]) if cps else np.zeros((0, 3))
    r = 0.02 * float(np.linalg.norm(V.max(0) - V.min(0)))
    scene = [trimesh.Trimesh(V, F, process=False)]
    scene[0].visual.vertex_colors = np.tile([200, 200, 200, 255], (len(V), 1))
    for g in Gm: scene.append(sphere(g, r, [0, 220, 0, 255]))     # YESIL = manufacturer CP
    for p in P: scene.append(sphere(p, r*0.8, [230, 0, 0, 255]))  # KIRMIZI = model tahmin
    out = f"results/w1_glb/WEI_{pid}.glb"
    trimesh.util.concatenate(scene).export(out)
    print(f"{pid}: {len(G)} manufacturer(yesil) | {len(P)} model(kirmizi) -> {out}")
print("\n-> results/w1_glb/*.glb  Windows 3D Viewer'da ac. KIRMIZI-only kureler = FP (model buldu, manufacturer listelemedi)")
print("-> Bak: kirmizilar gercek KABLO GIRISI mi (GT eksik) yoksa test-point/vida/montaj mi (model yanlis)?")
