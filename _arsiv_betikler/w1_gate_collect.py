# -*- coding: utf-8 -*-
"""URUN-SADIK gate-skor toplayici: robot_cp.extract yolunun AYNISI (4-model union + avg_probs wire-gate)
ama esik 0 -> butun cps'e wire_score yaz, TP/FP etiketle, diske dok. Hem WEI hem PXC held-out.
Sonra w1_gate_analyze.py grouped-CV ile per-mfg optimal esik bulur (in-sample degil, DURUST)."""
import os, sys, json, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh, wire_gate
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from robot_cp import _vote2

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open("cp_config.json"))
CKPTS = cfg["current_product"]["checkpoints"]
pp = cfg.get("prediction_postproc", {})
MV, VC, CL = int(pp.get("min_vertices", 30)), float(pp.get("vertex_confidence_mask", 0.5)), float(pp.get("cluster_mm", 5.0))
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 145
oos = set(open("pxc_out_of_scope.txt").read().split()) if os.path.exists("pxc_out_of_scope.txt") else set()
heldW = set(open("_hw_r3.txt").read().split())

models = [load_any(c, dev=dev)[:2] for c in CKPTS]
os.environ["BA_ALLOW_SEEN"] = "1"

# parca listeleri: WEI = _hw_r3 held-out; PXC = eligible in-scope, deterministik ornek
allp = list(eligible())
weis = [(m, p, jf, s) for m, p, jf, s in allp if m == "WEI" and p in heldW]
pxcs = [(m, p, jf, s) for m, p, jf, s in allp if m == "PXC" and p not in oos]
random.seed(0); random.shuffle(pxcs); pxcs = pxcs[:N_PER]
parts = weis[:N_PER] + pxcs
print(f"toplanacak: {len([1 for m,*_ in parts if m=='WEI'])} WEI + {len([1 for m,*_ in parts if m=='PXC'])} PXC = {len(parts)}", flush=True)


def derive(V, F, pb):
    return cp_openings.connection_points(V, F, pb.argmax(-1), min_v=MV, classes=(CE, CT), dedupe_mm=10.0,
                                         probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL)


rows = []; t0 = time.time()
for k, (mfg, pid, jf, stp) in enumerate(parts, 1):
    try:
        j = json.load(open(jf, encoding="utf-8-sig"))
        Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
        G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j["ConnectionPoints"]], float)
        Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j["ConnectionPoints"]], float)
        if not len(G): continue
        Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
        Vr, Fr = step_to_mesh(stp)
        V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000); V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
        per = []; acc = None
        for model, meta in models:
            _opd = f"{OP}_k{int(meta.get('k_eig', 64))}"
            _, pb = D.predict(model, meta, V, F, device=dev, op_cache_dir=_opd, return_probs=True)
            pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
            per.append(derive(V, F, pb))
        avg = acc / len(models)
        cps = _vote2(per, min_votes=1)                         # UNION (urun)
        if not cps: continue
        wire_gate.apply(V, F, avg, cps, CE, CT, threshold=0.0)  # sadece wire_score yaz
        R, t, _ = align_frames(Vr, Vj)
        P = np.array([np.asarray(c["point"]) for c in cps], float) @ R.T + t
        tol = max(3.0, 0.06 * float(np.linalg.norm(Vj.max(0) - Vj.min(0))))
        diff = P[:, None, :] - G[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
        pe = np.where(np.abs(al) <= 40.0, np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1), np.inf)
        yy = np.zeros(len(P), int); order = sorted((pe[a, b], a, b) for a in range(len(P)) for b in range(len(G)) if pe[a, b] <= tol)
        up, ug = set(), set()
        for dd, a, b in order:
            if a in up or b in ug: continue
            up.add(a); ug.add(b); yy[a] = 1
        for c, y in zip(cps, yy):
            rows.append({"mfg": mfg, "pid": pid, "ws": float(c["wire_score"]), "tp": int(y),
                         "votes": int(c.get("_votes", 1)), "N": int(len(G))})
    except Exception:
        continue
    if k % 20 == 0: print(f"  {k}/{len(parts)}  {len(rows)} pred  {time.time()-t0:.0f}s", flush=True)

json.dump(rows, open("results/gate_scores.json", "w"))
nW = len({r["pid"] for r in rows if r["mfg"] == "WEI"}); nP = len({r["pid"] for r in rows if r["mfg"] == "PXC"})
print(f"-> results/gate_scores.json  {len(rows)} pred, {nW} WEI + {nP} PXC parca  {time.time()-t0:.0f}s")
