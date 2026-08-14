# -*- coding: utf-8 -*-
"""Op 2 enabling: f1_sweep.extract'in POZISYONLU kopyasi. Standart havuzu (tum part) X+y+grp+mfg+ngt
YANINDA candidate POZISYONLARINI (JSON frame) da kaydet -> results/f1_pool_pos.npz. Deployed
f1_sweep_data.npz'ye DOKUNMAZ. Sonra op2_spatial.py offline spatial-rerank dener. Op-cache'ler sicak."""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh, connector3d, wire_gate
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from f1_sweep import union_all, HELD

NPZ = "results/f1_pool_pos.npz"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cks = json.load(open("cp_config.json"))["robot_vote2_checkpoints"]
    models = [load_any(c, dev=dev)[:2] for c in cks]
    os.environ["BA_ALLOW_SEEN"] = "1"
    parts = [p for p in eligible() if (p[0] == "WEI" and p[1] in HELD)] + [p for p in eligible() if p[0] == "PXC"]
    print(f"{len(parts)} part | pozisyonlu pool (GPU, sicak cache)", flush=True)
    _pp = json.load(open("cp_config.json")).get("prediction_postproc", {})
    MV = int(_pp.get("min_vertices", 30)); VC = float(_pp.get("vertex_confidence_mask", 0.5)); CL = float(_pp.get("cluster_mm", 5.0))
    Xs, votes, tp, grp, mfgs, Ps, ngt = [], [], [], [], [], [], {}
    t0 = time.time()
    for k, (mfg, pid, jf, stp) in enumerate(parts, 1):
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            if not len(G): continue
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            per = []; acc = None
            for model, meta in models:
                _, pb = D.predict(model, meta, V, F, device=dev, op_cache_dir=f"{OP}_k{int(meta.get('k_eig', 64))}", return_probs=True)
                pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
                per.append(cp_openings.connection_points(V, F, pb.argmax(-1), min_v=MV, classes=(CE, CT),
                           dedupe_mm=10.0, probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL))
            probs = acc / len(per); cps = union_all(per)
            if not cps: continue
            X = wire_gate.feats_for(V, F, probs, cps, CE, CT)
            R, t, _ = align_frames(Vr, Vj)
            P = np.array([np.asarray(c["point"]) for c in cps], float) @ R.T + t
            tol = max(3.0, 0.06 * float(np.linalg.norm(Vj.max(0) - Vj.min(0))))
            diff = P[:, None, :] - G[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.where(np.abs(al) <= 40.0, np.linalg.norm(diff - al[..., None]*Gd[None, :, :], axis=-1), np.inf)
            yy = np.zeros(len(P), int); order = sorted((pe[a, b], a, b) for a in range(len(P)) for b in range(len(G)) if pe[a, b] <= tol)
            up, ug = set(), set()
            for dd, a, b in order:
                if a in up or b in ug: continue
                up.add(a); ug.add(b); yy[a] = 1
            Xs.append(X); votes.append(np.array([c["_votes"] for c in cps])); tp.append(yy); Ps.append(P)
            grp += [k]*len(cps); mfgs += [1 if mfg == "WEI" else 0]*len(cps); ngt[k] = len(G)
        except Exception:
            continue
        if k % 25 == 0: print(f"  {k}/{len(parts)}  {time.time()-t0:.0f}s", flush=True)
    X = np.vstack(Xs); v = np.concatenate(votes); y = np.concatenate(tp); P = np.vstack(Ps)
    grp = np.array(grp); mfgs = np.array(mfgs)
    gid = np.array(sorted(ngt)); ng = np.array([ngt[g] for g in gid])
    np.savez(NPZ, X=X, votes=v, y=y, groups=grp, mfg=mfgs, grp_ids=gid, ngt=ng, P=P)
    print(f"  -> {NPZ} ({len(y)} union CP, pozisyonlu)  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
