# -*- coding: utf-8 -*-
"""DAL B (kendi-kosabilir): tel-girisi vs tool-agzi AYIRICI, YAPISAL ozelliklerle + otomatik etiketle.

recon: 4 global ozellik (size/depth/vote/conf) ayirmiyor (AUC<=0.61). Bu, FIZIKSEL/YAPISAL sinyalleri
dener: acikligin kutup-esligi (en yakin diger acikliga mesafe, komsu yogunlugu), disa-doğruluk (dis
kabukta mi ic govdede mi), bolge sekli (yuvarlak delik vs uzun yariq), bolge yogunlugu. Otomatik etiket:
uretici CP'ye eslesen = TEL (1), digeri = TOOL/fazla (0). PARCAYA GORE held-out (leakage yok) AUC.
AUC>0.75 -> cp_openings'e genel kapi. Aksi halde: insan wire/tool etiketi gerekir.
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh, connector3d
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from robot_cp import _vote2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HELD = set(open("_hw_r3.txt").read().split())


def feats_for(V, F, probs, cps):
    """her CP icin yapisal ozellik vektoru (mesh frame)."""
    conn = probs[:, CE] + probs[:, CT]
    ctr = 0.5 * (V.min(0) + V.max(0)); ext = V.max(0) - V.min(0)
    P = np.array([np.asarray(c["point"], float) for c in cps])
    Dv = np.array([np.asarray(c["direction"], float) for c in cps])
    Dv = Dv / (np.linalg.norm(Dv, axis=1, keepdims=True) + 1e-9)
    X = []
    for i, c in enumerate(cps):
        p = P[i]; d = Dv[i]
        area = float(c.get("area", 0.0)); size = 2.0 * (area / np.pi) ** 0.5 if area > 0 else 0.0
        depth = float(c.get("insertion_depth_mm", 0.0))
        # esleme: en yakin diger CP + 12mm ici komsu sayisi
        if len(P) > 1:
            dd = np.linalg.norm(P - p, axis=1); dd[i] = 1e9
            nn = float(dd.min()); nclose = int((dd <= 12.0).sum())
        else:
            nn = 50.0; nclose = 0
        # disa-dogruluk: kendi disa-isini uzerinde konumu
        u = (p - ctr); mo = np.abs((V - ctr) @ d).max()
        outward = float((u @ d) / (mo + 1e-9))
        # yerel bolge: p'nin 6mm cevresi
        near = np.linalg.norm(V - p, axis=1) <= 6.0
        nverts = int(near.sum())
        ce_frac = float(probs[near, CE].mean()) if near.any() else 0.0
        ct_frac = float(probs[near, CT].mean()) if near.any() else 0.0
        # bolge sekli: yakin verteks bulutunun PCA eksen oranlari (yuvarlak~1, yariq>>1)
        if near.sum() >= 6:
            Q = V[near] - V[near].mean(0)
            sv = np.linalg.svd(Q, compute_uv=False)
            aspect = float(sv[0] / (sv[1] + 1e-6)); flat = float(sv[2] / (sv[0] + 1e-6))
        else:
            aspect = 1.0; flat = 0.0
        # eksen-boyu derinlik profili: p'den d yonunde 15mm ici baglanti-verteks orani
        rel = V - p; al = rel @ d; perp = np.linalg.norm(rel - al[:, None] * d[None, :], axis=1)
        chan = (al >= 0) & (al <= 15) & (perp <= 4)
        chan_conn = float(conn[chan].mean()) if chan.any() else 0.0
        X.append([size, depth, nn, nclose, outward, nverts, ce_frac, ct_frac, aspect, flat, chan_conn,
                  float(c.get("_votes", 1)), float(c.get("confidence", 0.0))])
    return np.array(X, float)


FEAT_NAMES = ["size", "depth", "nn_dist", "n_close", "outward", "nverts", "ce_frac", "ct_frac",
              "aspect", "flat", "chan_conn", "votes", "conf"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cks = json.load(open("cp_config.json"))["robot_vote2_checkpoints"]
    models = [load_any(c, dev=dev)[:2] for c in cks]
    os.environ["BA_ALLOW_SEEN"] = "1"
    parts = [p for p in eligible() if (p[0] == "WEI" and p[1] in HELD)]
    parts += [p for p in eligible() if p[0] == "PXC"][:90]
    print(f"{len(parts)} parca | ozellik cikarimi", flush=True)

    Xall = []; yall = []; groups = []; mfgall = []; ngt = {}; t0 = time.time()
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
                _, pb = D.predict(model, meta, V, F, device=dev, op_cache_dir=OP, return_probs=True)
                pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
                per.append(cp_openings.connection_points(V, F, pb.argmax(-1), min_v=30, classes=(CE, CT),
                           dedupe_mm=10.0, probs=pb, vertex_conf=0.5, ct_depth_min_mm=1.0, cluster_mm=5.0))
            probs = acc / len(models)
            cps = _vote2(per)
            if not cps: continue
            X = feats_for(V, F, probs, cps)
            # otomatik etiket: mesh-frame CP -> JSON frame -> mfg CP eslesme
            R, t, _ = align_frames(Vr, Vj)
            P = np.array([np.asarray(c["point"]) for c in cps], float) @ R.T + t
            tol = max(3.0, 0.06 * float(np.linalg.norm(Vj.max(0) - Vj.min(0))))
            diff = P[:, None, :] - G[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
            y = np.zeros(len(P), int); order = sorted((pe[a, b], a, b) for a in range(len(P)) for b in range(len(G)) if pe[a, b] <= tol)
            up, ug = set(), set()
            for dd, a, b in order:
                if a in up or b in ug: continue
                up.add(a); ug.add(b); y[a] = 1
            Xall.append(X); yall.append(y); groups += [k] * len(y)
            mfgall += [1 if mfg == "WEI" else 0] * len(y); ngt[k] = len(G)
        except Exception:
            continue
        if k % 25 == 0: print(f"  {k}/{len(parts)}  {time.time()-t0:.0f}s", flush=True)

    X = np.vstack(Xall); y = np.concatenate(yall); groups = np.array(groups)
    print(f"\n=== AYIRICI ({len(y)} CP: {int(y.sum())} tel / {int((1-y).sum())} tool) ===")
    # PARCAYA GORE 5-fold held-out AUC (leakage yok)
    gkf = GroupKFold(n_splits=5); oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  YAPISAL AYIRICI held-out AUC = {auc:.3f}")
    print(f"  (recon global-ozellik en iyi 0.61 idi; >0.75 = cp_openings kapisi ise yarar)")
    # ozellik onemleri
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05).fit(X, y)
    imp = sorted(zip(FEAT_NAMES, clf.feature_importances_), key=lambda z: -z[1])
    print("  en onemli ozellikler:", ", ".join(f"{n}={v:.2f}" for n, v in imp[:6]))
    json.dump({"auc": float(auc), "n": len(y), "importances": dict(zip(FEAT_NAMES, clf.feature_importances_.tolist()))},
              open("results/wire_discriminator.json", "w"), indent=1)
    ngt_arr = np.array([ngt.get(g, 0) for g in sorted(ngt)]); grp_ids = np.array(sorted(ngt))
    np.savez("results/wire_discr_data.npz", X=X, y=y, groups=groups, mfg=np.array(mfgall),
             ngt=ngt_arr, grp_ids=grp_ids, feat_names=np.array(FEAT_NAMES))
    print("  -> results/wire_discr_data.npz (uctan-uca dogrulama icin)")


if __name__ == "__main__":
    main()
