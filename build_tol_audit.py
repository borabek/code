# -*- coding: utf-8 -*-
"""GOREV 2a: physical-tolerance audit verisi. Held-out arbiter (WEI+PXC, leakage-guard) her parca icin
aday pozisyon+boyut+wire_score + manufacturer CP pozisyon+yon KAYDET. Sonra tol_audit_sweep.py OFFLINE
farkli tolerans modlarini (current / opening_radius / size_scaled / wire_fit) dener -> ALL/WEI/PXC F1.
Amac: mevcut tolerans FIZIKSELDEN sikiysa, dogru toleransla ALL 0.756 kalkar mi (metrik duzeltmesi)."""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh, connector3d, wire_gate
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from robot_cp import _vote2
from f1_sweep import HELD

dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results/tol_audit_pool.json"


def main():
    cks = json.load(open("cp_config.json"))["robot_vote2_checkpoints"]
    models = [load_any(c, dev=dev)[:2] for c in cks]
    _pp = json.load(open("cp_config.json")).get("prediction_postproc", {})
    MV, VC, CL = int(_pp.get("min_vertices", 30)), float(_pp.get("vertex_confidence_mask", 0.5)), float(_pp.get("cluster_mm", 5.0))
    parts = [(m, p, jf, stp) for m, p, jf, stp in eligible() if (m == "WEI" and p in HELD) or m == "PXC"]
    print(f"{len(parts)} held-out arbiter parca", flush=True)
    gate = wire_gate._load()
    out = {}; t0 = time.time()
    for k, (mfg, pid, jf, stp) in enumerate(parts, 1):
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            if not len(G): continue
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000); V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            per = []; acc = None
            for model, meta in models:
                _, pb = D.predict(model, meta, V, F, device=dev, op_cache_dir=f"{OP}_k{int(meta.get('k_eig',64))}", return_probs=True)
                pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
                per.append(cp_openings.connection_points(V, F, pb.argmax(-1), min_v=MV, classes=(CE, CT), dedupe_mm=10.0, probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL))
            avg = acc/len(per); cps = _vote2(per)
            if not cps: continue
            Xf = wire_gate.feats_for(V, None, avg, cps, CE, CT)
            ws = gate["clf"].predict_proba(Xf)[:, 1] if gate else np.ones(len(cps))
            R, t, _ = align_frames(Vr, Vj)
            P = np.array([np.asarray(c["point"]) for c in cps], float) @ R.T + t
            size = np.array([2.0*(float(c.get("area", 0.0))/np.pi)**0.5 if c.get("area", 0) > 0 else 0.0 for c in cps])
            diag = float(np.linalg.norm(Vj.max(0)-Vj.min(0)))
            # her (aday a, uretici b): dik mesafe + eksen mesafe
            diff = P[:, None, :] - G[None, :, :]; al = (diff*Gd[None, :, :]).sum(-1)
            perp = np.linalg.norm(diff - al[..., None]*Gd[None, :, :], axis=-1)
            out[pid] = {"mfg": mfg, "diag": diag, "N": int(len(G)), "ws": ws.tolist(), "size": size.tolist(),
                        "perp": perp.tolist(), "along": al.tolist()}   # perp/along: (n_cand x n_gt)
        except Exception:
            continue
        if k % 40 == 0: print(f"  {k}/{len(parts)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(OUT, "w"))
    print(f"-> {OUT} ({len(out)} parca)  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
