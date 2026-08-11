# -*- coding: utf-8 -*-
"""KARAR: TUM WEI tipik parcalarda agresif aday uretimi candidate oracle'i 0.826 ustune cikariyor mu?
baseline (6k, 4 model) vs agresif (6k+9k+12k, 7 model, dusuk vc/min_v). reach = GT CP'ye tol icinde
aday var mi. Cikarsa (>=0.88) WEI 0.80 mumkun; kalirsa (~0.83) segmentasyon susuyor = etiket.
Bu SEFER 4 degil TUM WEI tipik parca (onceki probe yetersizdi)."""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh, connector3d
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from robot_cp import _vote2
from f1_sweep import HELD

dev = "cuda" if torch.cuda.is_available() else "cpu"
HICP = 11


def derive(V, F, pb, mv, vc, cl):
    return cp_openings.connection_points(V, F, pb.argmax(-1), min_v=mv, classes=(CE, CT), dedupe_mm=10.0,
                                         probs=pb, vertex_conf=vc, ct_depth_min_mm=1.0, cluster_mm=cl)


def reach(cps, Vr, Vj, G, Gd):
    if not cps or not len(G): return np.zeros(len(G), bool)
    R, t, _ = align_frames(Vr, Vj)
    P = np.array([np.asarray(c["point"]) for c in cps], float) @ R.T + t
    tol = max(3.0, 0.06 * float(np.linalg.norm(Vj.max(0) - Vj.min(0))))
    diff = P[:, None, :] - G[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
    pe = np.where(np.abs(al) <= 40.0, np.linalg.norm(diff - al[..., None]*Gd[None, :, :], axis=-1), np.inf)
    return (pe.min(0) <= tol)


def main():
    prod = json.load(open("cp_config.json"))["robot_vote2_checkpoints"]
    extra = ["results/seg_extra/recall_hard_keig128_s0.pt", "results/seg_extra/recall_hard_keig128_s1.pt", "results/seg_extra/recall_hard_keig128_s2.pt"]
    m4 = [load_any(c, dev=dev)[:2] for c in prod]
    m7 = m4 + [load_any(c, dev=dev)[:2] for c in extra]
    os.environ["BA_ALLOW_SEEN"] = "1"
    wei = []
    for mfg, pid, jf, stp in eligible():
        if mfg != "WEI" or pid not in HELD: continue
        try:
            n = len(json.load(open(jf, encoding="utf-8-sig")).get("ConnectionPoints", []))
            if 1 <= n < HICP: wei.append((pid, jf, stp))   # WEI tipik = non-high-CP
        except Exception: pass
    print(f"{len(wei)} WEI tipik parca | baseline(6k/4mdl) vs agresif(6k+9k+12k/7mdl)", flush=True)

    bt = at = GT = 0; t0 = time.time()
    for k, (pid, jf, stp) in enumerate(wei, 1):
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j["ConnectionPoints"]], float)
            Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j["ConnectionPoints"]], float)
            if not len(G): continue
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vr, Fr = step_to_mesh(stp)

            def runres(models, target, params):
                try:
                    Vt, Ft = thesis_remesh.remesh_uniform(Vr, Fr, target=target)
                    Vt = np.ascontiguousarray(Vt, np.float64); Ft = np.ascontiguousarray(Ft, np.int64)
                except Exception: return []
                cs = []
                for model, meta in models:
                    try:
                        _, pb = D.predict(model, meta, Vt, Ft, device=dev, op_cache_dir=f"{OP}_k{int(meta.get('k_eig',64))}_{target}", return_probs=True)
                        pb = np.asarray(pb, float)
                        for mv, vc, cl in params: cs.append(derive(Vt, Ft, pb, mv, vc, cl))
                    except Exception: pass
                return _vote2([c for c in cs if c])

            base = runres(m4, 6000, [(30, 0.5, 5.0)])
            aggr = base + runres(m7, 6000, [(10, 0.20, 3.0)]) + runres(m7, 9000, [(8, 0.10, 0.0), (4, 0.05, 0.0)]) + runres(m7, 12000, [(6, 0.10, 0.0)])
            rb = reach(base, Vr, Vj, G, Gd); ra = reach(aggr, Vr, Vj, G, Gd)
            bt += rb.sum(); at += ra.sum(); GT += len(G)
        except Exception:
            continue
        if k % 20 == 0: print(f"  {k}/{len(wei)}  base {bt/max(GT,1):.3f} aggr {at/max(GT,1):.3f}  {time.time()-t0:.0f}s", flush=True)
    print(f"\nWEI TIPIK candidate oracle ({GT} CP): baseline {bt/max(GT,1):.3f} -> agresif {at/max(GT,1):.3f}  (+{(at-bt)/max(GT,1):.3f})")
    print("KARAR: " + ("agresif WEI oracle >=0.88 -> WEI 0.80 MUMKUN, selector'a gec" if at/max(GT,1) >= 0.88 else
                        "agresif WEI oracle <0.88 -> segmentasyon susuyor, WEI 0.80 = ETIKET (145 parcada kanitli)"))


if __name__ == "__main__":
    main()
