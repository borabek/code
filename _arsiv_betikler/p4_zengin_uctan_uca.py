# -*- coding: utf-8 -*-
"""P4: ZENGIN BLOKLAR UCTAN UCA -- karar burada verilir.

P2 aday duzeyinde: konum9 +0.0114, cokyaricap24 +0.0169, hepsi +0.0231 (ucu de HER IKI
ureticide pozitif). Aday duzeyi bu arastirmada ALTI KEZ yaniltti; bu yuzden karar uctan uca,
kilitli olcum kumesinde (194 parca / 171 grup), GRUP bootstrap ile, `karar_olcutu` uzerinden.

EGITIM VERISI: results/zengin_parite.npz (1599 parca; parite-sadik adaylar, votes<=4)
OLCUM: results/_der_zengin.pkl (ayni adaylar + 33 zengin sutun)

DONUSUM: dagitilan yapinin parca-ici z-skoru her kolda uygulanir (urunle ayni).

KILL: karar_olcutu -- tanidik >= -0.01, bolme ORTALAMASI >= +0.01, en kotu bolme kotulesmesin,
hicbir bolmede > 0.05 kayip, VE GA ile KANITLI kazanc.
"""
import collections
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ZEN = "results/zengin_parite.npz"
DERZ = "results/_der_zengin.pkl"


def main():
    import karar_olcutu
    import olcum_kumesi
    import wire_gate
    from big_arbiter import eligible
    from sina_kume import esle, f1w
    from sklearn.ensemble import RandomForestClassifier

    mfg_of = {p: m for m, p, jf, s in eligible()}
    DER, rap = olcum_kumesi.kume(DERZ)
    olcum_kumesi.rapor_bas(rap)
    for r in DER:
        r["mfg"] = mfg_of.get(r["pid"], "?")
    gk = olcum_kumesi.geo_anahtarlari()
    tg = {r["geo"] for r in DER}

    d = np.load(ZEN, allow_pickle=True)
    X22 = np.asarray(d["X22"], float); XR = np.asarray(d["XR"], float)
    ytr = np.asarray(d["y"]); tpid = np.array([str(x) for x in d["pids"]])
    tmfg = np.array([str(x) for x in d["mfg"]])
    tgrp = np.array([gk.get(p, "yok:" + p) for p in tpid])
    hepsi = ~np.isin(tgrp, list(tg))
    print(f"egitim: {len(ytr)} aday / {len(np.unique(tpid))} parca | "
          f"sizinti disi {int(hepsi.sum())}", flush=True)

    dag = wire_gate._load(wire_gate.MODEL_PATH)
    DON = dag.get("donusum")
    kod = {k: collections.Counter(mfg_of.get(p, "?") for p in tpid[tmfg == k]).most_common(1)[0][0]
           for k in np.unique(tmfg)}

    KOL = {"A 22 temel": None, "B +konum9": list(range(0, 9)),
           "C +cokyaricap24": list(range(9, 33)), "D +33 hepsi": list(range(XR.shape[1]))}

    def donustur(X, pidler):
        if not DON:
            return X
        Z = np.zeros((len(X), X.shape[1] * 2))
        for u in np.unique(pidler):
            i = np.where(pidler == u)[0]
            Z[i] = wire_gate.parca_ici(X[i], DON)
        return Z

    BOLME = [("tanidik", hepsi, DER)]
    for k, mad in kod.items():
        alt = [x for x in DER if x["mfg"] == mad]
        if len(alt) >= 10:
            BOLME.append((mad, (tmfg != k) & hepsi, alt))

    SON, PARCA = {}, {}
    print(f"\n{'kol':<18}" + "".join(f"{b:>12}" for b, _, _ in BOLME) + f"{'robot':>10}")
    for ad, sec in KOL.items():
        Xt = X22 if sec is None else np.hstack([X22, XR[:, sec]])
        Mt = donustur(Xt, tpid)
        SON[ad] = {}; sat = f"{ad:<18}"
        for b, keep, alt in BOLME:
            clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                         random_state=0).fit(Mt[keep], ytr[keep])
            m = {"clf": clf, "n_feat": Mt.shape[1], "donusum": DON}
            det, rob = [], []
            for r in alt:
                P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
                if r["X"] is not None and r.get("XR") is not None:
                    Xr = r["X"] if sec is None else np.hstack([r["X"], r["XR"][:, sec]])
                    s = wire_gate.karar_skoru(m, Xr)
                    k2 = wire_gate.karar_maskesi(s)
                    if k2.any():
                        P = r["P"][k2]; Pd = r["Pd"][k2]
                rj = "cok" if r["n"] >= 8 else "dusuk"
                det.append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
                rob.append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
            SON[ad][b] = float(f1w(det))
            PARCA[(ad, b)] = (det, rob, [x["geo"] for x in alt])
            sat += f"{f1w(det):>12.4f}"
        sat += f"{f1w(PARCA[(ad,'tanidik')][1]):>10.4f}"
        print(sat, flush=True)

    def ga(a, b_):
        out = {}
        for bb, _, _ in BOLME:
            if bb == "tanidik":
                continue
            da, _, g = PARCA[(a, bb)]; db, _, _ = PARCA[(b_, bb)]
            cift = list(zip(da, db))
            fn = lambda rows: f1w([y for _, y in rows]) - f1w([x for x, _ in rows])
            _, lo, hi = olcum_kumesi.grup_bootstrap(cift, g, fn, n=2000)
            out[bb] = (lo, hi)
        return out

    print("\n=== KARAR ===")
    gecen = {}
    for ad in KOL:
        if ad.startswith("A"):
            continue
        k = karar_olcutu.degerlendir(SON["A 22 temel"], SON[ad], ga=ga("A 22 temel", ad))
        gecen[ad] = bool(k)
        print(f"\nA -> {ad}:  {k}")
    kazanan = max((a for a in gecen if gecen[a]),
                  key=lambda a: SON[a]["tanidik"], default=None)
    print(f"\nSONUC: {kazanan + ' DAGITILABILIR' if kazanan else 'HICBIRI GECMEDI'}")
    with io.open("results/p4_zengin_uctan_uca.json", "w", encoding="utf-8") as f:
        json.dump({"kollar": SON, "gecen": gecen, "kazanan": kazanan,
                   "robot": {a: float(f1w(PARCA[(a, "tanidik")][1])) for a in KOL}}, f, indent=1)
    with open("results/p4_parca.pkl", "wb") as f:
        pickle.dump(PARCA, f)
    print("makbuz -> results/p4_zengin_uctan_uca.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
