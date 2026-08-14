# -*- coding: utf-8 -*-
"""P2: B-rep silindir kapilarini DOGRU olcumlere gore yeniden ayarla.

NEDEN: `_fit_circle` duzeltmesi silindir yaricaplarini ~3.5 kat buyuttu ve eksen noktasini
gercekten eksene tasidi (metinle 327/327, medyan hata 0.0000mm). Ama asagi akistaki iki kapi
CARPIK degerlere gore ayarlanmisti:
  r_range ust siniri 12mm -> carpik olcekte gercekte ~42mm demekti (fiilen filtresiz).
                             Duzeltmeden SONRA 12mm gercekten 12mm; buyuk agizlar ELENIYOR.
  max_off_mm 5.0        -> eksen noktasi yaydan ~r kadar kaymisti; 5.0 o kaymayi tolere
                             ediyordu. Nokta artik eksende; 5.0 fazla gevsek olabilir
                             (komsu deliklere eslesme riski).
OLCULEN ETKI (DEV, duzeltme oncesi->sonrasi): tespit 0.7073 -> 0.7089 (+0.0016) ama
robot-hazir 0.4974 -> 0.4841 (-0.0133). Yani duzeltme DOGRU, kapilar BAYAT.

KILL: hicbir ayar duzeltme-oncesi robot-hazir degerini (0.4974) gecmezse kapilar eski
degerlerinde birakilir ve duzeltme yalnizca size_mm dogrulugu icin tutulur.
Karar DEV'de verilir, VAL'de SINANIR (kilitli harcanmaz).
"""
import os, sys, json, pickle, itertools
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import importlib
    import cp_openings, robot_cp, wire_gate
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from sklearn.ensemble import RandomForestClassifier
    from j_konum_ortalama import vote_avg
    from sina_kume import esle, f1w, pr

    kume = (sys.argv[1] if len(sys.argv) > 1 else "dev").lower()
    cf = f"results/_probs_{kume}.pkl"
    if not os.path.exists(cf) and kume == "dev":
        cf = "results/_h_probs.pkl"
    cache = pickle.load(open(cf, "rb"))
    for r in cache:
        r.setdefault("pid", os.path.basename(r["stp"]).split("_")[1])
    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"]); VC = float(pp["vertex_confidence_mask"]); CL = float(pp["cluster_mm"])
    THR = {"dusuk": float(cfg["robot_wire_gate_threshold"]),
           "cok": float(cfg["robot_wire_gate_threshold_highcp"])}

    d = np.load("results/gate_regrow_data_rt2.npz", allow_pickle=True)
    gk = json.load(open("results/_strict_geometry_keys.json"))
    tg = {gk.get(r["pid"], "yok:" + r["pid"]) for r in cache}
    pids = np.array([str(x) for x in d["pids"]])
    keep = ~np.isin(np.array([gk.get(p, "yok:" + p) for p in pids]), list(tg))
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(d["X"][keep], d["y"][keep])
    print(f"KUME={kume} | {len(cache)} parca | gate {int(keep.sum())} aday", flush=True)

    def kos(max_off, r_max):
        cp_openings.BREP_MAX_OFF = float(max_off)
        cp_openings.BREP_R_MAX = float(r_max)
        det, rob = [], []
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
            P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
            if cps:
                Xc = wire_gate.feats_for(V, F, sum(plist) / len(plist), cps, CE, CT)
                m = clf.predict_proba(Xc)[:, 1] >= (THR["cok"] if is_hi else THR["dusuk"])
                if m.any():
                    P = np.array([c["point"] for c, k in zip(cps, m) if k], float)
                    Pd = np.array([c["direction"] for c, k in zip(cps, m) if k], float)
            k = "cok" if r["n"] >= 8 else "dusuk"
            det.append((k,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
            rob.append((k,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
        return det, rob

    print(f"\n{'max_off':>8}{'r_max':>8}{'tespit':>9}{'ROBOT':>9}{'kesin':>8}{'recall':>8}")
    R = {}
    combos = [(5.0, 12.0), (5.0, 20.0), (5.0, 40.0), (3.0, 20.0), (3.0, 40.0),
              (2.0, 20.0), (2.0, 40.0), (1.5, 40.0)]
    for mo, rm in combos:
        det, rob = kos(mo, rm)
        R[(mo, rm)] = (det, rob)
        p_, r_ = pr(det)
        print(f"{mo:>8.1f}{rm:>8.1f}{f1w(det):>9.4f}{f1w(rob):>9.4f}{p_:>8.3f}{r_:>8.3f}",
              flush=True)

    taban = R[(5.0, 12.0)]
    en_iyi = max(R, key=lambda k: f1w(R[k][1]))
    rng = np.random.RandomState(0)
    n = len(taban[0]); IX = [rng.randint(0, n, n) for _ in range(2000)]
    print(f"\nEN IYI (robot-hazir): max_off={en_iyi[0]} r_max={en_iyi[1]}")
    out = {"taban": {"tespit": f1w(taban[0]), "robot": f1w(taban[1])},
           "en_iyi_ayar": list(en_iyi),
           "en_iyi": {"tespit": f1w(R[en_iyi][0]), "robot": f1w(R[en_iyi][1])},
           "tum": {f"{k[0]}|{k[1]}": {"tespit": f1w(v[0]), "robot": f1w(v[1])}
                   for k, v in R.items()}}
    for mi, mn in ((0, "tespit"), (1, "robot")):
        a, b = R[en_iyi][mi], taban[mi]
        ds = np.array([f1w([a[i] for i in ix]) - f1w([b[i] for i in ix]) for ix in IX])
        lo_, hi_ = np.percentile(ds, [2.5, 97.5])
        print(f"  {mn:<8}{ds.mean():>+9.4f}  [{lo_:+.4f}, {hi_:+.4f}]  "
              f"{'BELIRGIN' if lo_ > 0 or hi_ < 0 else 'gurultu'}")
        out.setdefault("bootstrap", {})[mn] = [float(ds.mean()), float(lo_), float(hi_)]
    print(f"\nKILL: duzeltme-oncesi robot-hazir 0.4974'u gecmezse kapilar ESKI degerlerde kalir -> "
          f"{f1w(R[en_iyi][1]):.4f} => {'AYARLA' if f1w(R[en_iyi][1]) > 0.4974 else 'BIRAK'}")
    json.dump(out, open(f"results/p2_brep_kapi_{kume}.json", "w"), indent=1)
    print(f"makbuz -> results/p2_brep_kapi_{kume}.json")


if __name__ == "__main__":
    main()
