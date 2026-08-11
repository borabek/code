# -*- coding: utf-8 -*-
"""FAZ 2 / P1 -- GIZLI MODEL SINYALI (DiffusionNet vertex embedding'leri).
Fikir: gate su ana kadar SADECE 5-sinif olasiligini ve el-yapimi geometriyi gordu. Modelin
son-katman-oncesi 128-dim vertex temsili cok daha zengin -- hic kullanilmadi.
Cikarim (hook: last_lin girdisi):
  - parca basina 128-dim vertex embedding (birincil model recall_hard_s2; embedding uzaylari
    modeller arasi hizali OLMADIGI icin TEK model kullanilir -- ortalama almak yanlis olurdu)
  - AsIRI-UYUM kontrolu: 128-dim'i global PCA ile 24'e indir (fit SADECE train-fold'da degil,
    unsupervised oldugu icin tum vertexlerden ornekle -- etiket kullanilmaz, sizinti yok)
  - aday basina: 6mm bolge mean+max (48) | 3/6/12/24mm halka mean (96) | global parca mean (24)
                 + insert-channel olasilik profili (5 sinif x 4 derinlik = 20)
Cikti: results/embed_feats.npz (EMB, part_ids ile hizali)  -> p2_p1_eval.py ile GO/NO-GO."""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh, wire_gate
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from robot_cp import _vote2

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open("cp_config.json"))
CK = cfg["current_product"]["checkpoints"]
pp_ = cfg.get("prediction_postproc", {})
MV, VC, CL = int(pp_.get("min_vertices", 30)), float(pp_.get("vertex_confidence_mask", 0.5)), float(pp_.get("cluster_mm", 5.0))
RINGS = [(0.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 24.0)]
CHAN_D = (0.0, 5.0, 10.0, 15.0)
NPCA = 24


class EmbedTap:
    """last_lin'in GIRDISINI yakala = son-katman-oncesi vertex temsili (C_width=128)."""
    def __init__(self, model):
        self.buf = None
        self.h = model.last_lin.register_forward_hook(lambda m, i, o: setattr(self, "buf", i[0].detach()))

    def close(self): self.h.remove()


def embed_predict(model, meta, V, F, tap, op_dir):
    _, probs = D.predict(model, meta, V, F, device=dev, op_cache_dir=op_dir, return_probs=True)
    e = tap.buf.cpu().numpy()
    if e.ndim == 3: e = e[0]                     # (1,Nv,128) -> (Nv,128)
    return np.asarray(probs, float), e


def main():
    models = [load_any(c, dev=dev)[:2] for c in CK]
    prim_model, prim_meta = models[0]
    tap = EmbedTap(prim_model)
    oos = set(open("pxc_out_of_scope.txt").read().split()) if os.path.exists("pxc_out_of_scope.txt") else set()
    held = set(open("_hw_r3.txt").read().split())
    def _filt(l): return [(m, p, jf, s) for m, p, jf, s in l
                          if (m == "WEI" and p in held) or (m == "PXC" and p not in oos)]
    clean_ids = {p for _, p, _, _ in _filt(eligible())}
    os.environ["BA_ALLOW_SEEN"] = "1"
    parts = _filt(eligible())
    print(f"{len(parts)} parca | embedding dim {prim_model.last_lin.in_features} -> PCA {NPCA}", flush=True)

    # --- 1. gecis: PCA'yi fit etmek icin vertex embedding ornekle (ETIKET KULLANILMAZ) ---
    samp = []
    for mfg, pid, jf, stp in parts[::12]:
        try:
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            _, emb = embed_predict(prim_model, prim_meta, V, F, tap, f"{OP}_k{int(prim_meta.get('k_eig',64))}")
            samp.append(emb[::20])
        except Exception:
            continue
        if len(samp) >= 60: break
    S = np.vstack(samp)
    mu = S.mean(0); Sc = S - mu
    U, sv, Vt = np.linalg.svd(Sc, full_matrices=False)
    W = Vt[:NPCA].T                                    # (128, NPCA)
    print(f"  PCA fit: {S.shape} ornek | aciklanan varyans {100*(sv[:NPCA]**2).sum()/(sv**2).sum():.1f}%", flush=True)

    EMB, PIDS, GG, NGT, SEENL = [], [], [], {}, []
    t0 = time.time()
    for k, (mfg, pid, jf, stp) in enumerate(parts, 1):
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j["ConnectionPoints"]], float)
            if not len(G): continue
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            per = []; acc = None; emb = None
            for mi, (model, meta) in enumerate(models):
                opd = f"{OP}_k{int(meta.get('k_eig',64))}"
                if mi == 0:
                    pb, e = embed_predict(model, meta, V, F, tap, opd); emb = e
                else:
                    _, pb2 = D.predict(model, meta, V, F, device=dev, op_cache_dir=opd, return_probs=True)
                    pb = np.asarray(pb2, float)
                acc = pb if acc is None else acc + pb
                per.append(cp_openings.connection_points(V, F, pb.argmax(-1), min_v=MV, classes=(CE, CT),
                           dedupe_mm=10.0, probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL))
            probs = acc / len(per)
            cps = _vote2(per, min_votes=1)
            if not cps: continue
            Z = (emb - mu) @ W                                   # (Nv, NPCA)
            gz = Z.mean(0)
            feats = []
            for c in cps:
                p = np.asarray(c["point"], float); dirv = np.asarray(c["direction"], float)
                dirv = dirv / (np.linalg.norm(dirv) + 1e-9)
                rel = V - p; dist = np.linalg.norm(rel, axis=1)
                f = []
                m6 = dist <= 6.0
                f += list(Z[m6].mean(0)) if m6.any() else [0.0] * NPCA          # bolge mean
                f += list(Z[m6].max(0)) if m6.any() else [0.0] * NPCA           # bolge max
                for lo, hi in RINGS:                                            # halkalar
                    mr = (dist > lo) & (dist <= hi)
                    f += list(Z[mr].mean(0)) if mr.any() else [0.0] * NPCA
                f += list(gz)                                                   # global parca
                al = rel @ dirv; perp = np.linalg.norm(rel - al[:, None] * dirv[None, :], axis=1)
                for t in CHAN_D:                                                # insert-channel profili
                    ch = (np.abs(al - t) <= 2.5) & (perp <= 4.0)
                    f += list(probs[ch].mean(0)) if ch.any() else [0.0] * probs.shape[1]
                feats.append(f)
            gi = len(NGT); NGT[gi] = len(G)
            EMB.append(np.array(feats, float)); GG += [gi] * len(cps)
            PIDS.append(pid); SEENL.append(0 if pid in clean_ids else 1)
        except Exception:
            continue
        if k % 25 == 0: print(f"  {k}/{len(parts)}  {len(NGT)} ok  {time.time()-t0:.0f}s", flush=True)
    tap.close()
    E = np.vstack(EMB)
    np.savez("results/embed_feats.npz", EMB=E, groups=np.array(GG), part_ids=np.array(PIDS),
             seen=np.array(SEENL), grp_ids=np.array(sorted(NGT)), ngt=np.array([NGT[g] for g in sorted(NGT)]),
             pca_W=W, pca_mu=mu)
    print(f"-> results/embed_feats.npz  {E.shape[0]} aday x {E.shape[1]} embedding-feature, "
          f"{len(NGT)} parca  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
