# -*- coding: utf-8 -*-
"""GERCEK CP-F1: ayni tahminler, SIKILASTIRILAN toleransla puanlanir.

NEDEN: proje boyunca eslesme toleransi max(3mm, %6 x parca kosegeni) ve eksenel pencere +-40mm.
Buyuk/yogun parcalarda bu cok gevsek: 249mm kosegenli 64-CP'li bir parcada tolerans 14.9mm
oluyor ve 14mm otedeki bir tahmin "dogru" sayilabiliyor. 5 parcalik sondada kesinlik
%6-tolerans ile 1.000, 2mm ile 0.455 cikti -- yani "sifir yanlis tespit" tolerans eseriydi.

Bu, projenin onceki sisme hatasinin KARDESI (eski "%98 kesinlik" alet agizlarini dogru
sayiyordu). Fark: orada TANIM yanlisti, burada TOLERANS gevsek.

TASARIM: cikarim BIR KEZ kosar, ayni CP'ler dort ayri toleransla puanlanir -- fark boylece
modelden degil YALNIZ olcutten gelir. Protokol P9 ile birebir: ayni 100 parca (tohum 202),
test aileleri gate egitiminden cikarilmis, esikler buyuk veride aile-disi OOF ile secilmis,
korpus-agirlikli (dusuk 0.895 / cok 0.105).
"""
import os, sys, json
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = {"dusuk": 0.895, "cok": 0.105}
NPZ = "results/gate_regrow_data_rt2.npz"

# CARPAN TASARIMI: dik tolerans ve eksenel pencere AYRI AYRI degisir.
# Ilk kosuda ikisini birlikte degistirmistim (5mm/+-20 -> 3mm/+-10) ve ucurumun hangisinden
# geldigi ayirt edilemedi. Iki degiskeni ayni anda oynatan bir deney hicbir seyi kanitlamaz.
PERP = [("%6xkosegen", lambda diag: max(3.0, 0.06 * diag)),
        ("5mm", lambda diag: 5.0), ("3mm", lambda diag: 3.0), ("2mm", lambda diag: 2.0)]
AXIAL = [40.0, 10.0]
CACHE = "results/_tolerans_cache6.pkl"


def big_thr(npz, test_fams):
    """Esikleri BUYUK veride, test ailelerine bakmadan sec (P8 protokolu)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    d = np.load(npz, allow_pickle=True)
    X, y, groups, fams = d["X"], d["y"], d["groups"], d["fams"]
    keep = ~np.isin(fams.astype(str), list(test_fams))
    X, y, groups, fams = X[keep], y[keep], groups[keep], fams[keep]
    ngt = dict(zip(d["grp_ids"].tolist(), d["ngt"].tolist()))
    reg = np.array([("cok" if int(ngt.get(int(g), 0)) >= 8 else "dusuk") for g in groups])
    tot = {"dusuk": 0, "cok": 0}
    for g in {int(g) for g in groups}:
        n = int(ngt.get(g, 0))
        if n > 0:
            tot["cok" if n >= 8 else "dusuk"] += n
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, fams.astype(str)):
        oof[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                         random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    out = {}
    for k in ("dusuk", "cok"):
        best = (0.0, 0.40)
        for thr in np.arange(0.20, 0.71, 0.05):
            m = reg == k
            sel = (oof >= thr) & m
            tp = int((y[sel] == 1).sum())
            fp = int(sel.sum()) - tp
            p = tp / max(tp + fp, 1)
            r = tp / max(tot[k], 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            if f1 > best[0]:
                best = (f1, float(thr))
        out[k] = best[1]
    return out


def main():
    import torch, thesis_remesh, cp_openings, robot_cp, wire_gate, diffusionnet as D
    from cad_eval import align_frames
    from infer_step_cp import step_to_mesh, load_any
    from big_arbiter import eligible
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from json_dataset import family_key
    from sklearn.ensemble import RandomForestClassifier

    cfg = json.load(open("cp_config.json"))
    pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"])
    VC = float(pp["vertex_confidence_mask"])
    CL = float(pp["cluster_mm"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LOCK = set(json.load(open("results/split_lock.json"))["locked_parts"])
    models = [load_any(c, dev=dev)[:2] for c in cfg["current_product"]["checkpoints"]]

    parts = []
    for m, p, jf, s in eligible():
        if p in LOCK:
            continue
        try:
            n = len(json.load(open(jf, encoding="utf-8-sig"))["ConnectionPoints"])
        except Exception:
            continue
        if n > 0:
            parts.append((m, p, jf, s, n))
    rng = np.random.RandomState(202)          # P9 ile AYNI tohum, AYNI bolum
    lo = [x for x in parts if x[4] < 8]
    hi = [x for x in parts if x[4] >= 8]
    sel = ([lo[i] for i in rng.choice(len(lo), 70, replace=False)] +
           [hi[i] for i in rng.choice(len(hi), 30, replace=False)])
    test_fams = {family_key(p[1]) for p in sel}
    print(f"{len(sel)} parca (70 dusuk / 30 cok), {len(test_fams)} aile gate egitiminden disarida",
          flush=True)

    thr = big_thr(NPZ, test_fams)
    print(f"esikler (buyuk veride aile-disi OOF): {thr}", flush=True)
    d = np.load(NPZ, allow_pickle=True)
    keep = ~np.isin(d["fams"].astype(str), list(test_fams))
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(d["X"][keep], d["y"][keep])
    print(f"gate {int(keep.sum())} aday ile egitildi", flush=True)

    import pickle
    if os.path.exists(CACHE):
        cache = pickle.load(open(CACHE, "rb"))
        print(f"onbellek diskten yuklendi: {len(cache)} parca (cikarim tekrar kosmuyor)", flush=True)
        return _report(cache)

    cache = []
    for mfg, pid, jf, stp, n in sel:
        try:
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64)
            F = np.ascontiguousarray(F, np.int64)
            pbs = []
            for model, meta in models:
                _, pb = D.predict(model, meta, V, F, device=dev,
                                  op_cache_dir=f"results/step_infer/ops_k{int(meta.get('k_eig', 64))}",
                                  return_probs=True)
                pbs.append(np.asarray(pb, float))
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[q[c] for c in "XYZ"] for q in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in j["ConnectionPoints"]], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in j["ConnectionPoints"]],
                          float)
            Gd /= np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9
            R, t, ares = align_frames(Vr, Vj)

            def der(pr):
                return [cp_openings.connection_points(
                    V, F, pb.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
                    probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
                    conn_promote=pr, step_path=stp) for pb in pbs]

            base = robot_cp._vote2(der(0.0), min_votes=1)
            is_hi = robot_cp._highcp_router(stp, V, len(base))
            cps = robot_cp._vote2(der(0.25), min_votes=1) if is_hi else base
            kept = []
            if cps:
                probs = sum(pbs) / len(pbs)
                sc = clf.predict_proba(wire_gate.feats_for(V, F, probs, cps, CE, CT))[:, 1]
                t_ = thr["cok"] if is_hi else thr["dusuk"]
                kept = [c for c, s_ in zip(cps, sc) if s_ >= t_]
            cache.append(dict(
                Q=np.array([c["point"] for c in kept], float) if kept else np.zeros((0, 3)),
                Qd=np.array([c["direction"] for c in kept], float) if kept else np.zeros((0, 3)),
                G=(G - t) @ R, Gd=Gd @ R, n=n,
                diag=float(np.linalg.norm(V.max(0) - V.min(0))), ares=float(ares),
                V=V.astype(np.float32), F=F.astype(np.int32),
                area=np.array([float(c.get("area", 0.0)) for c in kept], np.float32)))
        except Exception:
            continue
        if len(cache) % 25 == 0:
            print(f"  {len(cache)} parca", flush=True)
    print(f"onbellek {len(cache)} parca", flush=True)
    pickle.dump(cache, open(CACHE, "wb"))
    return _report(cache)


def _report(cache):
    import numpy as np

    def score(tolfn, axial):
        agg = {"dusuk": [0, 0, 0], "cok": [0, 0, 0]}
        for r in cache:
            Q, G, Gd = r["Q"], r["G"], r["Gd"]
            tol = tolfn(r["diag"])
            hit = np.zeros(len(G), bool)
            used = set()
            if len(Q) and len(G):
                diff = Q[:, None, :] - G[None, :, :]
                al = (diff * Gd[None, :, :]).sum(-1)
                pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
                pe = np.where(np.abs(al) <= axial, pe, np.inf)
                for d_, a_, b_ in sorted((pe[a, b], a, b)
                                         for a in range(len(Q)) for b in range(len(G))):
                    if d_ > tol or a_ in used or hit[b_]:
                        continue
                    hit[b_] = True
                    used.add(a_)
            tp = int(hit.sum())
            k = "cok" if r["n"] >= 8 else "dusuk"
            agg[k][0] += tp
            agg[k][1] += len(Q) - tp
            agg[k][2] += len(G) - tp
        out = {}
        for k, (T, Fp, Fn) in agg.items():
            p = T / max(T + Fp, 1)
            rc = T / max(T + Fn, 1)
            out[k] = dict(f1=2 * p * rc / max(p + rc, 1e-9), p=p, r=rc, tp=T, fp=Fp, fn=Fn)
        out["w"] = sum(W[k] * out[k]["f1"] for k in W)
        return out

    print()
    print(f"{'dik tolerans':<14}{'eksenel':>9}{'dusuk-CP':>10}{'cok-CP':>9}"
          f"{'AGIRLIKLI':>11}{'kesinlik':>10}{'recall':>8}")
    res = {}
    for ax in AXIAL:
        for name, fn in PERP:
            r = score(fn, ax)
            res[f"{name} / +-{ax:.0f}mm"] = r
            tp = r["dusuk"]["tp"] + r["cok"]["tp"]
            fp = r["dusuk"]["fp"] + r["cok"]["fp"]
            fn_ = r["dusuk"]["fn"] + r["cok"]["fn"]
            print(f"{name:<14}{ax:>7.0f}mm{r['dusuk']['f1']:>10.4f}{r['cok']['f1']:>9.4f}"
                  f"{r['w']:>11.4f}{tp / max(tp + fp, 1):>10.3f}"
                  f"{tp / max(tp + fn_, 1):>8.3f}", flush=True)
        print()

    ares = [r["ares"] for r in cache]
    print(f"\nhizalama residual (STEP<->JSON): medyan {np.median(ares):.2f}mm  "
          f"%90 {np.percentile(ares, 90):.2f}mm  max {max(ares):.2f}mm")
    print("  -> sabit 2mm toleransin bir kismi HIZALAMA hatasidir, modelin degil")
    json.dump({"sonuc": res, "align_residual_mm": ares, "n_parca": len(cache),
               "protokol": "P9 ile birebir (tohum 202, 70 dusuk / 30 cok, aile-disi gate)"},
              open("results/tolerans_gercegi.json", "w"), indent=1, default=float)
    print("makbuz -> results/tolerans_gercegi.json")


if __name__ == "__main__":
    main()
