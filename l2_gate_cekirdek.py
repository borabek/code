# -*- coding: utf-8 -*-
"""L2: gate'i GENELLESEN cekirdek uzerine kur -- ezberleyen ozellikleri at.

K OLCTU (durust bolme): 13 ozellikten yalnizca `votes` gercekten transfer ediyor
(AUC 0.680 -> 0.662, dusus 0.018). Digerlerinin durust AUC'si 0.51-0.60, yani neredeyse
yazi-tura; ezber bolmesinde 0.68-0.77 gorunmelerinin sebebi geometrik ikizler.

AMA AUC URUN METRIGI DEGIL, ve K ayrica gosterdi ki TUM ozelliklerle RF'in durust AUC'si
(0.842) en yuksek. Yani "ezberliyor" tek basina atmak icin yeterli degil -- asil soru
operasyon esiginde UCTAN UCA F1 ve precision.

KILL: azaltilmis kume 13'u UCTAN UCA gecmezse duser (yani mevcut kume korunur).
"""
import os, sys, json, pickle
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = {"dusuk": 0.895, "cok": 0.105}


def main():
    import cp_openings, robot_cp, wire_gate
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from big_arbiter import eligible
    from j_konum_ortalama import vote_avg

    cfg = json.load(open("cp_config.json"))
    pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"]); VC = float(pp["vertex_confidence_mask"]); CL = float(pp["cluster_mm"])
    cache = pickle.load(open("results/_h_probs.pkl", "rb"))
    d = np.load("results/gate_regrow_data_rt2.npz", allow_pickle=True)
    X = d["X"]; y = d["y"]; groups = d["groups"]
    pids = np.array([str(x) for x in d["pids"]])
    ngt = dict(zip(d["grp_ids"].tolist(), d["ngt"].tolist()))
    gk = json.load(open("results/_geometry_keys.json"))
    names = wire_gate.FEAT_NAMES_13

    LOCK = set(json.load(open("results/split_lock.json"))["locked_parts"])
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
    rng = np.random.RandomState(202)
    lo = [x for x in parts if x[4] < 8]
    hi = [x for x in parts if x[4] >= 8]
    sel = ([lo[i] for i in rng.choice(len(lo), 70, replace=False)] +
           [hi[i] for i in rng.choice(len(hi), 30, replace=False)])
    tg = {gk.get(p[1], "yok:" + p[1]) for p in sel}
    Gg = np.array([gk.get(p, "yok:" + p) for p in pids])
    keep = ~np.isin(Gg, list(tg))
    reg_all = np.array([("cok" if int(ngt.get(int(g), 0)) >= 8 else "dusuk") for g in groups])

    # Aday turetme BIR KEZ (urun hattiyla); ozellik kumeleri sonra bedava taranir.
    DER = []
    for r in cache:
        V = np.ascontiguousarray(r["V"], np.float64)
        F = np.ascontiguousarray(r["F"], np.int64)
        plist = [np.asarray(pb, np.float64) for pb in r["pbs"]]
        der = [cp_openings.connection_points(
            V, F, pb.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
            probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
            step_path=r["stp"]) for pb in plist]
        base = vote_avg(der, min_votes=1, mode="wmean")
        is_hi = robot_cp._highcp_router(r["stp"], V, len(base))
        if is_hi:
            der2 = [cp_openings.connection_points(
                V, F, pb.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
                probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
                conn_promote=0.25, step_path=r["stp"]) for pb in plist]
            cps = vote_avg(der2, min_votes=1, mode="wmean")
        else:
            cps = base
        Xc = None
        if cps:
            probs = sum(plist) / len(plist)
            Xc = wire_gate.feats_for(V, F, probs, cps, CE, CT)
        DER.append(dict(X=Xc, is_hi=is_hi,
                        P=np.array([c["point"] for c in cps], float) if cps else np.zeros((0, 3)),
                        Pd=np.array([c["direction"] for c in cps], float) if cps else np.zeros((0, 3)),
                        G=r["G"], Gd=r["Gd"], n=r["n"], diag=r["diag"]))
    print(f"{len(DER)} parca turetildi", flush=True)

    def evaluate(cols, lab):
        Xk = X[keep][:, cols]; yk = y[keep]; gg = Gg[keep]; reg = reg_all[keep]
        tot = {"dusuk": 0, "cok": 0}
        for g in {int(g) for g in groups[keep]}:
            n = int(ngt.get(g, 0))
            if n > 0:
                tot["cok" if n >= 8 else "dusuk"] += n
        o = np.zeros(len(yk))
        for tr, te in GroupKFold(n_splits=5).split(Xk, yk, gg):
            o[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                           random_state=0).fit(Xk[tr], yk[tr]).predict_proba(Xk[te])[:, 1]
        thr = {}
        for k in ("dusuk", "cok"):
            b = (0.0, 0.40)
            for t in np.arange(0.20, 0.71, 0.05):
                m = reg == k
                s_ = (o >= t) & m
                tp = int((yk[s_] == 1).sum()); fp = int(s_.sum()) - tp
                p_ = tp / max(tp + fp, 1); r_ = tp / max(tot[k], 1)
                f = 2 * p_ * r_ / max(p_ + r_, 1e-9)
                if f > b[0]:
                    b = (f, float(t))
            thr[k] = b[1]
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                     random_state=0).fit(Xk, yk)
        out = {}
        for key, tol, am, pct in (("det", 0.0, 180.0, True), ("rob", 2.0, 10.0, False)):
            agg = {"dusuk": [0, 0, 0], "cok": [0, 0, 0]}
            for r in DER:
                if r["X"] is None or not len(r["P"]):
                    P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
                else:
                    sc = clf.predict_proba(r["X"][:, cols])[:, 1]
                    m = sc >= (thr["cok"] if r["is_hi"] else thr["dusuk"])
                    P = r["P"][m]; Pd = r["Pd"][m]
                G, Gd = r["G"], r["Gd"]
                hit = np.zeros(len(G), bool)
                used = set()
                if len(P) and len(G):
                    diff = P[:, None, :] - G[None, :, :]
                    al = (diff * Gd[None, :, :]).sum(-1)
                    pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
                    an = np.degrees(np.arccos(np.clip(np.abs(Pd @ Gd.T), 0, 1)))
                    tt = max(3.0, 0.06 * r["diag"]) if pct else tol
                    pe = np.where((np.abs(al) > 40) | (an > am), np.inf, pe)
                    for d_, a_, b_ in sorted((pe[a, b], a, b)
                                             for a in range(len(P)) for b in range(len(G))):
                        if d_ > tt or a_ in used or hit[b_]:
                            continue
                        hit[b_] = True
                        used.add(a_)
                tp = int(hit.sum())
                k = "cok" if r["n"] >= 8 else "dusuk"
                agg[k][0] += tp; agg[k][1] += len(P) - tp; agg[k][2] += len(G) - tp
            o_ = {}; TP = FP = FN = 0
            for k, (T, Fp, Fn) in agg.items():
                p_ = T / max(T + Fp, 1); rc = T / max(T + Fn, 1)
                o_[k] = 2 * p_ * rc / max(p_ + rc, 1e-9)
                TP += T; FP += Fp; FN += Fn
            out[key] = (sum(W[k] * o_[k] for k in W), TP / max(TP + FP, 1), TP / max(TP + FN, 1))
        print(f"{lab:<34}{out['det'][0]:>9.4f}{out['rob'][0]:>9.4f}"
              f"{out['det'][1]:>9.3f}{out['det'][2]:>8.3f}", flush=True)
        return out

    idx = {n: i for i, n in enumerate(names)}
    print()
    print(f"{'ozellik kumesi':<34}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>8}")
    # EZBER SIRALAMASI (K olcumu, AUC dususu buyukten kucuge)
    order = ["depth", "size", "aspect", "outward", "flat", "nn_dist", "ce_frac",
             "chan_conn", "ct_frac", "conf", "nverts", "n_close", "votes"]
    evaluate(list(range(13)), "13 ozellik (budamasiz)")
    for k in (3, 5, 7, 9):
        drop = set(order[:k])
        cols = [i for i in range(13) if names[i] not in drop]
        evaluate(cols, f"en ezberci {k} atildi ({13-k} kalir)")
    print()
    print("KILL: azaltilmis kume 13'u UCTAN UCA gecmezse duser.")


if __name__ == "__main__":
    main()
