# -*- coding: utf-8 -*-
"""Q6: 18-sutunlu (fiziksel ozellikli) gate UCTAN UCA kazaniyor mu?

Q5 OLCTU (aday duzeyi, grup-capraz OOF, 200 ayrik parca): 13 -> 13+5 fiziksel ozellik
OOF AUC 0.8576 -> 0.8996, aday F1 0.6997 -> 0.7554 (+0.0556); brep_r ile mevcut size
korelasyonu -0.009 (kopya degil, YENI bilgi).

AMA ADAY-DUZEYI F1 UCTAN UCA F1 DEGILDIR. E maddesi (eksen guveni) gate duzeyinde +0.0068
kazanip uctan uca -0.0125 KAYBETMISTI. Kabul karari yalnizca uctan uca verilir.

UC KOL -- iki degiskeni AYIRMAK sart. `_fiz` verisi hem YENI OZELLIKLERI hem GUNCEL HATTI
tasiyor ve guncel hat TEK BASINA DEV'de -0.0146 kaybettirmisti (gate v6 olcumu). Bu yuzden
`_fiz` verisinin ILK 13 SUTUNU ucuncu kol olarak kosulur:
    rt2 13  : dagitilan gate      (eski hat, 13 ozellik)
    fiz13   : guncel hat, 13 sutun -> VERI etkisini izole eder
    fiz18   : guncel hat, 18 sutun -> OZELLIK etkisini izole eder
Ikisini ayirmadan "18 sutun kazandi/kaybetti" demek iki etkiyi karistirmak olurdu.

KURAL 2 (yazili): iki gate ancak AYNI aday havuzunda karsilastirilir. Bu betik adaylari BIR KEZ
turetir ve 18 sutunu bir kez hesaplar; 13-sutunlu kollar ayni matrisin ilk 13 sutununu kullanir.

KILL (onceden yazili): DEV'de uctan uca tespit F1 artmazsa dagitilmaz. Artarsa VAL'de SINANIR;
VAL'de kaybederse yine dagitilmaz (uclu bolme kurali).
"""
import os, sys, json, pickle
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"          # test tarafinda 18 sutun uretilsin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KAYNAK = {"rt2 13": ("results/gate_regrow_data_rt2.npz", 13),
          "fiz13": ("results/gate_regrow_data_fiz.npz", 13),
          "fiz18": ("results/gate_regrow_data_fiz.npz", 18)}


def main():
    import cp_openings, robot_cp, wire_gate
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from sklearn.ensemble import RandomForestClassifier
    from j_konum_ortalama import vote_avg
    from sina_kume import esle, f1w, pr

    assert len(wire_gate.FEAT_NAMES) == 18, f"18 sutun bekleniyordu, {len(wire_gate.FEAT_NAMES)}"
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

    gk = json.load(open("results/_strict_geometry_keys.json"))
    tg = {gk.get(r["pid"], "yok:" + r["pid"]) for r in cache}
    CLF = {}
    for tag, (f, ncol) in KAYNAK.items():
        d = np.load(f, allow_pickle=True)
        pids = np.array([str(x) for x in d["pids"]])
        keep = ~np.isin(np.array([gk.get(p, "yok:" + p) for p in pids]), list(tg))
        Xt = d["X"][keep][:, :ncol]
        CLF[tag] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                          random_state=0).fit(Xt, d["y"][keep])
        print(f"gate {tag}: {int(keep.sum())} aday x {ncol} sutun", flush=True)

    # --- turetme BIR KEZ, 18 sutun bir kez ---
    DER = []
    for r in cache:
        V = np.ascontiguousarray(r["V"], np.float64)
        F = np.ascontiguousarray(r["F"], np.int64)
        plist = [np.asarray(pb, np.float64) for pb in r["pbs"]]
        mk = lambda pr_, **kw: cp_openings.connection_points(
            V, F, pr_.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
            probs=pr_, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
            step_path=r["stp"], **kw)
        # MERGE=urun -> urunun kendi birlestiricisi (_vote2: BENZERSIZ oy + agirlikli konum).
        # Varsayilan vote_avg, onceki olcumlerle karsilastirilabilirlik icin korunuyor.
        merge = ((lambda L: robot_cp._vote2(L, min_votes=1))
                 if os.environ.get("MERGE") == "urun"
                 else (lambda L: vote_avg(L, min_votes=1, mode="wmean")))
        base = merge([mk(pb) for pb in plist])
        is_hi = robot_cp._highcp_router(r["stp"], V, len(base))
        cps = merge([mk(pb, conn_promote=0.25) for pb in plist]) if is_hi else base
        Xc = (wire_gate.feats_for(V, F, sum(plist) / len(plist), cps, CE, CT,
                                  step_path=r["stp"]) if cps else None)
        DER.append(dict(X=Xc, is_hi=is_hi,
                        P=np.array([c["point"] for c in cps], float) if cps else np.zeros((0, 3)),
                        Pd=np.array([c["direction"] for c in cps], float) if cps else np.zeros((0, 3)),
                        G=r["G"], Gd=r["Gd"], n=r["n"], diag=r["diag"]))
    nad = sum(len(r["P"]) for r in DER)
    print(f"{len(DER)} parca | {nad} aday (AYNI havuz, uc kol icin de)", flush=True)

    def kos(tag):
        clf = CLF[tag]; nc = KAYNAK[tag][1]
        det, rob = [], []
        for r in DER:
            P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
            if r["X"] is not None:
                sc = clf.predict_proba(r["X"][:, :nc])[:, 1]
                m = sc >= (THR["cok"] if r["is_hi"] else THR["dusuk"])
                if m.any():
                    P = r["P"][m]; Pd = r["Pd"][m]
            k = "cok" if r["n"] >= 8 else "dusuk"
            det.append((k,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
            rob.append((k,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
        return det, rob

    print(f"\n{'gate':<24}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}")
    R = {}
    for tag, lab in (("rt2 13", "rt2 13 (dagitilan)"),
                     ("fiz13", "guncel hat, 13 sutun"),
                     ("fiz18", "guncel hat, 18 sutun")):
        det, rob = kos(tag)
        R[tag] = (det, rob)
        p_, r_ = pr(det)
        print(f"{lab:<24}{f1w(det):>9.4f}{f1w(rob):>9.4f}{p_:>9.3f}{r_:>9.3f}", flush=True)

    # parca-basina sayimlari SAKLA: havuzlanmis (DEV+VAL) manset ve GA bunlardan hesaplanir.
    # F1 dogrusal olmadigi icin iki kumenin F1 ORTALAMASI havuzlanmis F1'e esit DEGILDIR.
    _sfx = "_urun" if os.environ.get("MERGE") == "urun" else ""
    pickle.dump(R, open(f"results/q6_parca_{kume}{_sfx}.pkl", "wb"))
    rng = np.random.RandomState(0)
    n = len(R["rt2 13"][0])
    IX = [rng.randint(0, n, n) for _ in range(2000)]
    out = {}
    print("\nESLI BOOTSTRAP -- iki etki AYRI:")
    for lab, a_, b_ in (("OZELLIK (fiz18-fiz13)", "fiz18", "fiz13"),
                        ("VERI    (fiz13-rt2)  ", "fiz13", "rt2 13"),
                        ("TOPLAM  (fiz18-rt2)  ", "fiz18", "rt2 13")):
        for mi, mn in ((0, "tespit"), (1, "robot")):
            a, b = R[a_][mi], R[b_][mi]
            ds = np.array([f1w([a[i] for i in ix]) - f1w([b[i] for i in ix]) for ix in IX])
            lo_, hi_ = np.percentile(ds, [2.5, 97.5])
            print(f"  {lab} {mn:<7}{ds.mean():>+9.4f}  [{lo_:+.4f}, {hi_:+.4f}]  "
                  f"{'BELIRGIN' if lo_ > 0 or hi_ < 0 else 'gurultu'}", flush=True)
            out[f"{a_}-{b_}|{mn}"] = [float(ds.mean()), float(lo_), float(hi_)]

    dd = f1w(R["fiz18"][0]) - f1w(R["rt2 13"][0])
    print(f"\nKILL: uctan uca tespit artmazsa DAGITILMAZ -> toplam {dd:+.4f} => "
          f"{'DEVAM (VAL sinavi)' if dd > 0 else 'DAGITMA'}")
    json.dump({"kume": kume, "n_aday": int(nad),
               "tespit": {k: float(f1w(v[0])) for k, v in R.items()},
               "robot": {k: float(f1w(v[1])) for k, v in R.items()},
               "kesinlik": {k: float(pr(v[0])[0]) for k, v in R.items()},
               "recall": {k: float(pr(v[0])[1]) for k, v in R.items()},
               "bootstrap": out}, open(f"results/q6_fiz_uctan_uca_{kume}.json", "w"), indent=1)
    print(f"makbuz -> results/q6_fiz_uctan_uca_{kume}.json")


if __name__ == "__main__":
    main()
