# -*- coding: utf-8 -*-
"""Q5: B-rep fiziksel ozellikleri gate'e ARTIMLI bir sey katiyor mu?

Q4 OLCTU: `brep_r` AUC 0.707 (TP medyan 1.80mm, FP medyan 0.15mm), `bos_derinlik` 0.290
(ters yonde 0.710), `esesenli` 0.603 -- ucu de null'un (p95 ~0.52) cok uzerinde.

AMA BU YETMEZ: gate'te ZATEN `size` (mesh agiz genisligi) ve `depth` (isin sondaji) var.
Yeni ozellikler onlarin daha temiz bir kopyasiysa artimli degeri SIFIR olabilir. Tek basina
yuksek AUC, "yeni bilgi" demek DEGILDIR.

BU BETIK: ayni adaylarda 13 mevcut ozellik vs 13+5 fiziksel, GRUP-CAPRAZ (parca bazli) OOF ile
  - OOF AUC (bag duzeltmeli Mann-Whitney)
  - urun esiginde OOF F1 / kesinlik / recall
karsilastirir. Gruplama parca kimligi -> ayni parcanin adaylari hem egitimde hem testte olmaz.

DUZELTME (Q4'te kusurluydu): `bos_derinlik` "isin hic carpmadi" ile "hemen carpti" durumlarini
ayni 0.0 degerinde birlestiriyordu; burada carpmama -1 ile ayriliyor.

KILL: 18 ozellik 13'u OOF F1'de gecmezse kanal KAPANIR (AUC farki YETMEZ -- urun metrigi sart).
"""
import os, sys, json, pickle
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIZ = ["brep_r", "esesenli", "r_orani", "bos_derinlik", "gecen"]


def fiz_ozellik(p, d, cyl, mesh):
    import cp_geometry as G
    C, A, R = cyl
    o = dict(brep_r=0.0, esesenli=0, r_orani=1.0, bos_derinlik=-1.0, gecen=0)
    if len(C):
        rel = p - C
        al = (rel * A).sum(1)
        off = np.linalg.norm(rel - al[:, None] * A, axis=1)
        ok = off <= 3.0
        if ok.any():
            cand = np.where(ok)[0]
            j = cand[int(np.argmax(np.abs(A[cand] @ d)))]
            o["brep_r"] = float(R[j])
            par = np.abs(A @ A[j]) > 0.995
            dd = C - C[j]
            coax = par & (np.linalg.norm(dd - (dd @ A[j])[:, None] * A[j], axis=1) < 0.5)
            rr = R[coax]
            o["esesenli"] = int(coax.sum())
            if len(rr) > 1:
                o["r_orani"] = float(rr.max() / max(rr.min(), 1e-6))
    try:
        h = np.asarray(G.ray_hits(mesh, p + 0.05 * d, d, max_mm=200.0), float).ravel()
        if len(h):
            o["bos_derinlik"] = float(h[0])       # ilk "taban"a mesafe
            o["gecen"] = int(len(h) <= 1)
        else:
            o["bos_derinlik"] = -1.0              # HIC carpmadi: 0.0'dan AYRI tutulur
            o["gecen"] = 1
    except Exception:
        pass
    return o


def auc_mw(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.empty(len(x), float); o = np.argsort(x, kind="mergesort"); xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    import cp_openings, robot_cp, wire_gate, trimesh
    import brep_axes as B
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from j_konum_ortalama import vote_avg

    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"]); VC = float(pp["vertex_confidence_mask"]); CL = float(pp["cluster_mm"])
    THR = {"dusuk": float(cfg["robot_wire_gate_threshold"]),
           "cok": float(cfg["robot_wire_gate_threshold_highcp"])}

    X13, XF, Y, GRP, HI = [], [], [], [], []
    for kume in ("dev", "val"):
        cf = f"results/_probs_{kume}.pkl"
        if not os.path.exists(cf) and kume == "dev":
            cf = "results/_h_probs.pkl"
        cache = pickle.load(open(cf, "rb"))
        for r in cache:
            r.setdefault("pid", os.path.basename(r["stp"]).split("_")[1])
        for r in cache:
            V = np.ascontiguousarray(r["V"], np.float64)
            F = np.ascontiguousarray(r["F"], np.int64)
            plist = [np.asarray(pb, np.float64) for pb in r["pbs"]]
            mk = lambda pr_, **kw: cp_openings.connection_points(
                V, F, pr_.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
                probs=pr_, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
                step_path=r["stp"], **kw)
            merge = lambda L: vote_avg(L, min_votes=1, mode="wmean")
            base = merge([mk(pb) for pb in plist])
            is_hi = robot_cp._highcp_router(r["stp"], V, len(base))
            cps = merge([mk(pb, conn_promote=0.25) for pb in plist]) if is_hi else base
            if not cps:
                continue
            try:
                cyl = B.cylinders(r["stp"])
            except Exception:
                continue
            mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
            Xg = wire_gate.feats_for(V, F, sum(plist) / len(plist), cps, CE, CT)
            P = np.array([c["point"] for c in cps], float)
            D = np.array([c["direction"] for c in cps], float)
            G_, Gd = r["G"], r["Gd"]
            lab = np.zeros(len(P), bool)
            if len(G_):
                diff = P[:, None, :] - G_[None, :, :]
                al = (diff * Gd[None, :, :]).sum(-1)
                pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
                pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
                t0 = max(3.0, 0.06 * r["diag"])
                us, ug = set(), set()
                for dv, a_, b_ in sorted((pe[a, b], a, b)
                                         for a in range(len(P)) for b in range(len(G_))):
                    if dv > t0 or a_ in us or b_ in ug:
                        continue
                    us.add(a_); ug.add(b_); lab[a_] = True
            for k in range(len(P)):
                f = fiz_ozellik(P[k], D[k], cyl, mesh)
                X13.append(Xg[k]); XF.append([f[n] for n in FIZ])
                Y.append(bool(lab[k])); GRP.append(r["pid"]); HI.append(is_hi)
        print(f"  {kume} bitti ({len(Y)} aday)", flush=True)

    X13 = np.array(X13, float); XF = np.array(XF, float)
    Y = np.array(Y, bool); GRP = np.array(GRP); HI = np.array(HI, bool)
    X18 = np.hstack([X13, XF])
    print(f"\n{len(Y)} aday | TP {int(Y.sum())} | {len(set(GRP))} parca grubu")

    # mevcut `size` ve `depth` ile karsilastir -- yeni ozellik onlarin kopyasi mi?
    nm = list(wire_gate.FEAT_NAMES_13)
    print(f"\nTEK OZELLIK AUC (yeni bilgi mi, kopyasi mi):")
    for n in ("size", "depth"):
        print(f"  MEVCUT {n:<12}{auc_mw(X13[:, nm.index(n)], Y):>8.3f}")
    for i, n in enumerate(FIZ):
        print(f"  YENI   {n:<12}{auc_mw(XF[:, i], Y):>8.3f}")
    # dogrudan korelasyon: brep_r vs size
    c = np.corrcoef(XF[:, 0], X13[:, nm.index("size")])[0, 1]
    print(f"  -> brep_r ile size korelasyonu: {c:+.3f} "
          f"({'BAGIMSIZ' if abs(c) < 0.7 else 'KOPYA RISKI'})")

    def oof(X):
        o = np.zeros(len(Y))
        for tr, te in GroupKFold(n_splits=5).split(X, Y, GRP):
            o[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                           random_state=0).fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]
        return o

    print(f"\n{'gate':<22}{'OOF AUC':>10}{'F1':>9}{'kesin':>9}{'recall':>9}")
    res = {}
    for lab, X in (("13 mevcut", X13), ("13 + 5 fiziksel", X18)):
        o = oof(X)
        thr = np.where(HI, THR["cok"], THR["dusuk"])
        s = o >= thr
        tp = int((Y & s).sum()); fp = int((~Y & s).sum()); fn = int((Y & ~s).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        f1 = 2 * p_ * r_ / max(p_ + r_, 1e-9)
        a = auc_mw(o, Y)
        print(f"{lab:<22}{a:>10.4f}{f1:>9.4f}{p_:>9.3f}{r_:>9.3f}", flush=True)
        res[lab] = {"auc": float(a), "f1": float(f1), "kesinlik": float(p_), "recall": float(r_)}

    d = res["13 + 5 fiziksel"]["f1"] - res["13 mevcut"]["f1"]
    da = res["13 + 5 fiziksel"]["auc"] - res["13 mevcut"]["auc"]
    print(f"\nARTIMLI: AUC {da:+.4f} | aday-duzeyi F1 {d:+.4f}")
    print(f"KILL: aday-duzeyi F1 artmazsa kanal KAPANIR -> {'AC' if d > 0 else 'KAPAT'}")
    json.dump(res | {"artimli_f1": float(d), "artimli_auc": float(da),
                     "brep_r_size_korelasyon": float(c)},
              open("results/q5_artimli_deger.json", "w"), indent=1)
    np.savez("results/q5_ozellikler.npz", X13=X13, XF=XF, Y=Y, GRP=GRP, HI=HI)
    print("makbuz -> results/q5_artimli_deger.json (+ q5_ozellikler.npz)")


if __name__ == "__main__":
    main()
