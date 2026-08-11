# -*- coding: utf-8 -*-
"""ABB KOLU -- ogrenme egrisi tavanlanmadigi icin (hala +0.010 AUC/adim) VERI ekliyoruz.
Kaynak: Desktop/JSON'daki 386 ABB parcasi = 11,228 uretici CP (mevcut 1888 GT'nin 6 KATI),
ortalama 29 CP/parca -> tam da modelin coktugu HIGH-CP rejimi.
AVANTAJ: mesh JSON Graphic3d'den gelir ve CP'ler AYNI FRAME'dedir -> align_frames YOK,
hizalama hata kaynagi tamamen ortadan kalkar.
Cikti: results/abb_feats.npz (X13, XR, y, groups, part_ids, ngt) -- rich_feats ile ayni format."""
import os, sys, json, glob, time
import numpy as np, torch, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh, wire_gate
from infer_step_cp import load_any
from big_arbiter import CE, CT, OP
from robot_cp import _vote2
from build_rich_feats import rich_feats

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open("cp_config.json"))
CK = cfg["current_product"]["checkpoints"]
pp_ = cfg.get("prediction_postproc", {})
MV, VC, CL = int(pp_.get("min_vertices", 30)), float(pp_.get("vertex_confidence_mask", 0.5)), float(pp_.get("cluster_mm", 5.0))
MFG = sys.argv[1] if len(sys.argv) > 1 else "ABB"
LIM = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def main():
    models = [load_any(c, dev=dev)[:2] for c in CK]
    fs = sorted(glob.glob(f"C:/Users/DE00024082/Desktop/JSON/{MFG}.*.json"))
    if LIM: fs = fs[:LIM]
    print(f"{len(fs)} {MFG} parca | JSON mesh (ayni frame, hizalama YOK)", flush=True)
    X13, XR, YY, GG, POS, NGT, PIDS = [], [], [], [], [], {}, []
    t0 = time.time()
    for k, f in enumerate(fs, 1):
        try:
            j = json.load(open(f, encoding="utf-8-sig"))
            pts = j["Graphic3d"]["Points"]; idxs = j["Graphic3d"].get("Indices", [])
            if not idxs: continue
            V0 = np.array([[p["X"], p["Y"], p["Z"]] for p in pts], float)
            F0 = np.array(idxs, int).reshape(-1, 3)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j["ConnectionPoints"]], float)
            Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j["ConnectionPoints"]], float)
            if not len(G): continue
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            V, F = thesis_remesh.remesh_uniform(V0, F0, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            per = []; acc = None
            for model, meta in models:
                _, pb = D.predict(model, meta, V, F, device=dev,
                                  op_cache_dir=f"{OP}_k{int(meta.get('k_eig',64))}_abb", return_probs=True)
                pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
                per.append(cp_openings.connection_points(V, F, pb.argmax(-1), min_v=MV, classes=(CE, CT),
                           dedupe_mm=10.0, probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL))
            probs = acc / len(per)
            cps = _vote2(per, min_votes=1)
            if not cps: continue
            Nrm = trimesh.Trimesh(V, F, process=False).vertex_normals.view(np.ndarray)
            x13 = wire_gate.feats_for(V, F, probs, cps, CE, CT)
            xr = rich_feats(V, F, probs, cps, Nrm)
            P = np.array([np.asarray(c["point"]) for c in cps], float)     # AYNI FRAME -- donusum YOK
            tol = max(3.0, 0.06 * float(np.linalg.norm(V0.max(0) - V0.min(0))))
            diff = P[:, None, :] - G[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.where(np.abs(al) <= 40.0, np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1), np.inf)
            yy = np.zeros(len(P), int); up, ug = set(), set()
            for dd, a, b in sorted((pe[a, b], a, b) for a in range(len(P)) for b in range(len(G)) if pe[a, b] <= tol):
                if a in up or b in ug: continue
                up.add(a); ug.add(b); yy[a] = 1
            gi = len(NGT); NGT[gi] = len(G)
            X13.append(x13); XR.append(xr); YY.append(yy); POS.append(P); GG += [gi] * len(cps)
            PIDS.append(os.path.basename(f).replace(".json", ""))
        except Exception:
            continue
        if k % 25 == 0: print(f"  {k}/{len(fs)}  {len(NGT)} ok  {time.time()-t0:.0f}s", flush=True)
    y = np.concatenate(YY); gid = np.array(sorted(NGT)); ng = np.array([NGT[g] for g in gid])
    out = f"results/{MFG.lower()}_feats.npz"
    np.savez(out, X13=np.vstack(X13), XR=np.vstack(XR), y=y, groups=np.array(GG),
             pos=np.vstack(POS), grp_ids=gid, ngt=ng, part_ids=np.array(PIDS))
    print(f"-> {out}  {len(y)} aday, {len(NGT)} parca, GT {int(ng.sum())}, "
          f"aday-tavan recall {y.sum()/max(ng.sum(),1):.3f}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
