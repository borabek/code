# -*- coding: utf-8 -*-
"""T13: icbukey kenar topolojisi TP/FP ayirt ediyor mu? (madde 6, birinci kapi)

Olcum sirasi (kisayol yok): bu betik ADAY DUZEYINDE AUC + null testi yapar. Gecmezse tam
korpus uretimi ACILMAZ (1.5 saat tasarruf).

KILL (onceden yazili): hicbir ozellik null'un uzerinde AUC >= 0.60 vermezse kanal KAPANIR.
Ayrica ARTIMLI deger sart: 18 mevcut sutunun uzerine ne katiyor (grup-capraz OOF).
"""
import os, sys, json, pickle
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_uretici_disi import auc_mw


def main():
    import cp_openings, robot_cp, wire_gate, topo_feats
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"]); VC = float(pp["vertex_confidence_mask"]); CL = float(pp["cluster_mm"])
    ORAN = float(cfg.get("gate_goreli_oran", 0.5)); TABAN = float(cfg.get("gate_goreli_taban", 0.25))

    X18, XT, Y, GRP = [], [], [], []
    for kume in ("dev", "val"):
        cf = f"results/_probs_{kume}.pkl"
        if not os.path.exists(cf) and kume == "dev":
            cf = "results/_h_probs.pkl"
        cache = pickle.load(open(cf, "rb"))
        for r in cache:
            r.setdefault("pid", os.path.basename(r["stp"]).split("_")[1])
        for i, r in enumerate(cache, 1):
            V = np.ascontiguousarray(r["V"], np.float64)
            F = np.ascontiguousarray(r["F"], np.int64)
            plist = [np.asarray(pb, np.float64) for pb in r["pbs"]]
            mk = lambda pr_, **kw: cp_openings.connection_points(
                V, F, pr_.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
                probs=pr_, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
                step_path=r["stp"], **kw)
            merge = lambda L: robot_cp._vote2(L, min_votes=1)
            base = merge([mk(pb) for pb in plist])
            is_hi = robot_cp._highcp_router(r["stp"], V, len(base))
            cps = merge([mk(pb, conn_promote=0.25) for pb in plist]) if is_hi else base
            if not cps:
                continue
            Xg = wire_gate.feats_for(V, F, sum(plist) / len(plist), cps, CE, CT,
                                     step_path=r["stp"])[:, :18]
            onb = topo_feats.icbukey_kenarlar(V, F)      # parca basina BIR kez
            P = np.array([c["point"] for c in cps], float)
            D = np.array([c["direction"] for c in cps], float)
            G, Gd = r["G"], r["Gd"]
            lab = np.zeros(len(P), bool)
            if len(G):
                diff = P[:, None, :] - G[None, :, :]
                a_ = (diff * Gd[None, :, :]).sum(-1)
                pe = np.linalg.norm(diff - a_[..., None] * Gd[None, :, :], axis=-1)
                pe = np.where(np.abs(a_) <= 40.0, pe, np.inf)
                t0 = max(3.0, 0.06 * r["diag"]); us, ug = set(), set()
                for dv, x, y_ in sorted((pe[a, b], a, b)
                                        for a in range(len(P)) for b in range(len(G))):
                    if dv > t0 or x in us or y_ in ug:
                        continue
                    us.add(x); ug.add(y_); lab[x] = True
            for k in range(len(P)):
                X18.append(Xg[k])
                XT.append(topo_feats.topo_ozellik(V, F, P[k], D[k], onbellek=onb))
                Y.append(bool(lab[k])); GRP.append(r["pid"])
            if i % 25 == 0:
                print(f"  {kume} {i}/{len(cache)}", flush=True)
    X18 = np.array(X18, float); XT = np.array(XT, float)
    Y = np.array(Y, bool); GRP = np.array(GRP)
    print(f"\n{len(Y)} aday | TP {int(Y.sum())} | {len(set(GRP))} parca")

    rng = np.random.RandomState(0)
    print(f"\n{'ozellik':<12}{'AUC':>8}{'null p95':>10}{'TP med':>10}{'FP med':>10}{'karar':>9}")
    out = {}
    for i, n in enumerate(topo_feats.ISIMLER):
        a = auc_mw(XT[:, i], Y)
        nl = np.array([auc_mw(XT[:, i], rng.permutation(Y)) for _ in range(300)])
        p95 = float(np.percentile(np.abs(nl - 0.5), 95) + 0.5)
        kar = "CANLI" if abs(a - 0.5) + 0.5 >= max(0.60, p95) else "-"
        print(f"{n:<12}{a:>8.3f}{p95:>10.3f}{np.median(XT[Y, i]):>10.3f}"
              f"{np.median(XT[~Y, i]):>10.3f}{kar:>9}")
        out[n] = {"auc": float(a), "null_p95": p95, "karar": kar}

    def oof(M):
        o = np.zeros(len(Y))
        for tr, te in GroupKFold(n_splits=5).split(M, Y, GRP):
            o[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                           random_state=0).fit(M[tr], Y[tr]).predict_proba(M[te])[:, 1]
        m = np.zeros(len(o), bool)
        for u in np.unique(GRP):
            i = GRP == u; v = o[i]
            m[i] = (v >= ORAN * max(v.max(), 1e-9)) & (v >= TABAN)
        tp = int((Y & m).sum()); fp = int((~Y & m).sum()); fn = int((Y & ~m).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        return auc_mw(o, Y), 2 * p_ * r_ / max(p_ + r_, 1e-9), p_, r_

    print(f"\n{'gate':<22}{'OOF AUC':>10}{'F1':>9}{'kesin':>9}{'recall':>9}")
    a1, f1_, p1, r1 = oof(X18)
    print(f"{'18 mevcut':<22}{a1:>10.4f}{f1_:>9.4f}{p1:>9.3f}{r1:>9.3f}")
    a2, f2_, p2, r2 = oof(np.hstack([X18, XT]))
    print(f"{'18 + 4 topoloji':<22}{a2:>10.4f}{f2_:>9.4f}{p2:>9.3f}{r2:>9.3f}")
    print(f"\nARTIMLI: AUC {a2-a1:+.4f} | aday-duzeyi F1 {f2_-f1_:+.4f}")
    canli = [n for n in topo_feats.ISIMLER if out[n]["karar"] == "CANLI"]
    print(f"KILL: AUC>=0.60 veren ozellik VE artimli F1 artisi yoksa KAPANIR -> "
          f"{'AC' if (canli and f2_ > f1_) else 'KAPAT'}  (canli: {canli})")
    np.savez("results/t13_topo.npz", X18=X18, XT=XT, Y=Y, GRP=GRP)
    json.dump(out | {"artimli_auc": float(a2 - a1), "artimli_f1": float(f2_ - f1_)},
              open("results/t13_topo_auc.json", "w"), indent=1)
    print("makbuz -> results/t13_topo_auc.json (+ .npz)")


if __name__ == "__main__":
    main()
