# -*- coding: utf-8 -*-
"""U3: PARCA-ICI ozellik normalizasyonu -- ureticiler arasi SIRALAMA yarisi icin.

NEDEN BU: gece tesihs "cokusun ~%42'si KALIBRASYON, ~%58'i SIRALAMA" demisti. Kalibrasyon
yarisini GORELI ESIK cozdu (WEI disarida 0.3150 -> 0.4736): skoru parca ICINDE karsilastirmak.
Sıralama yarisina uc tedavi denendi ve oldu (olcek-bagimsizlastirma, model sinifi, imzali sutun
atma). Denenmemis olan, TAM DA ISE YARAYAN FIKRIN OZELLIK DUZEYINDEKI HALI:

    goreli esik = SKORU parca icinde karsilastir  (kalibrasyonu duzeltti)
    bu deney    = OZELLIGI parca icinde karsilastir (siralamayi duzeltir mi?)

Fark onemli: t10 (olcek-bagimsizlastirma) mm sutunlarini PARCA CAPINA boluyordu -- bu, mutlak
olcegi kaldirir ama dagilim SEKLINI birakir. Parca-ici SIRA ise dagilimin kendisini kaldirir:
"bu aday bu parcadaki en derin 2. delik" ifadesi ureticiden bagimsizdir, "derinlik 4.2mm" degildir.

ARMLAR
  A  ham X (taban, dagitilan)
  B  X + parca-ici SIRA (sutun sayisi x2)
  C  yalniz parca-ici SIRA
  D  X + parca-ici Z-SKOR (sira degil, olcek: aykiri degerleri korur)

KARAR KURALI her armda ayni: dagitilan goreli esik (oran x parca-maks VE mutlak taban).

KILL (onceden yazili, BUYUKLUK dahil):
  gorulmemis uretici EN KOTU durumu >= +0.01 artmali VE tanidik >= -0.01 kalmali.
  Gecen arm UCTAN UCA dogrulanmadan DAGITILMAZ (aday duzeyi bu gece 3 kez yanildi).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NPZ = "results/gate_regrow_data_topo.npz"
NJOB = 4      # u1 arka planda calisiyor -- tum cekirdekleri kapma


def parca_ici_sira(X, pids):
    """Her parca icinde, her sutun icin [0,1] araligina normalize SIRA."""
    out = np.zeros_like(X, dtype=float)
    for u in np.unique(pids):
        i = np.where(pids == u)[0]
        if len(i) == 1:
            out[i] = 0.5           # tek aday: sira bilgisi YOK, notr
            continue
        sub = X[i]
        rk = np.argsort(np.argsort(sub, axis=0), axis=0).astype(float)
        out[i] = rk / (len(i) - 1)
    return out


def parca_ici_z(X, pids):
    """Her parca icinde sutun bazli z-skor (sd=0 ise 0)."""
    out = np.zeros_like(X, dtype=float)
    for u in np.unique(pids):
        i = np.where(pids == u)[0]
        sub = X[i]
        sd = sub.std(0)
        out[i] = np.where(sd > 1e-12, (sub - sub.mean(0)) / np.where(sd > 1e-12, sd, 1.0), 0.0)
    return out


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    ORAN = float(cfg["gate_goreli_oran"]); TABAN = float(cfg["gate_goreli_taban"])
    gk = json.load(open("results/_strict_geometry_keys.json"))

    d = np.load(NPZ, allow_pickle=True)
    X = np.asarray(d["X"], float); Y = np.asarray(d["y"]).astype(bool)
    pids = np.array([str(x) for x in d["pids"]])
    mfg = np.array([str(x) for x in d["mfg"]])
    geo = np.array([gk.get(p, "yok:" + p) for p in pids])
    print(f"{len(Y)} aday | {len(set(pids))} parca | {len(set(geo))} geometri | "
          f"pozitif {Y.mean():.4f}", flush=True)

    print("parca-ici donusumler hesaplaniyor...", flush=True)
    SIRA = parca_ici_sira(X, pids)
    Z = parca_ici_z(X, pids)
    ARM = {"A ham": X,
           "B ham+sira": np.hstack([X, SIRA]),
           "C yalniz sira": SIRA,
           "D ham+zskor": np.hstack([X, Z])}

    def karar(skor):
        """Dagitilan goreli esik: parca icinde oran x maks VE mutlak taban."""
        m = np.zeros(len(skor), bool)
        for u in np.unique(pids):
            i = pids == u; v = skor[i]
            m[i] = (v >= ORAN * max(v.max(), 1e-9)) & (v >= TABAN)
        return m

    def f1(mask, sel=None):
        y = Y if sel is None else Y[sel]
        m = mask if sel is None else mask[sel]
        tp = int((y & m).sum()); fp = int((~y & m).sum()); fn = int((y & ~m).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        return 2 * p_ * r_ / max(p_ + r_, 1e-9)

    rf = lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                        n_jobs=NJOB, random_state=0)
    print(f"\n{'arm':<16}{'tanidik':>10}{'mfg0-disi':>11}{'mfg1-disi':>11}{'EN KOTU':>10}")
    sonuc = {}
    for ad, M in ARM.items():
        o = np.zeros(len(Y))
        for tr, te in GroupKFold(n_splits=5).split(M, Y, geo):
            o[te] = rf().fit(M[tr], Y[tr]).predict_proba(M[te])[:, 1]
        tan = f1(karar(o))
        dis = {}
        for k in np.unique(mfg):
            te = mfg == k
            s = np.zeros(len(Y))
            s[te] = rf().fit(M[~te], Y[~te]).predict_proba(M[te])[:, 1]
            dis[k] = f1(karar(s), te)
        ek = min(dis.values())
        print(f"{ad:<16}{tan:>10.4f}{dis['0']:>11.4f}{dis['1']:>11.4f}{ek:>10.4f}", flush=True)
        sonuc[ad] = {"tanidik": float(tan), "mfg0_disi": float(dis["0"]),
                     "mfg1_disi": float(dis["1"]), "en_kotu": float(ek)}

    t = sonuc["A ham"]
    print(f"\nKARAR (taban A: tanidik {t['tanidik']:.4f} | en kotu {t['en_kotu']:.4f})")
    kazanan = None
    for ad, s in sonuc.items():
        if ad.startswith("A"):
            continue
        dt = s["tanidik"] - t["tanidik"]; dk = s["en_kotu"] - t["en_kotu"]
        gecti = (dk >= 0.01) and (dt >= -0.01)
        print(f"  {ad:<16} tanidik {dt:+.4f} | en kotu {dk:+.4f} -> "
              f"{'GECTI' if gecti else 'GECMEDI'}")
        if gecti and (kazanan is None or s["en_kotu"] > sonuc[kazanan]["en_kotu"]):
            kazanan = ad
    print(f"\nSONUC: {(kazanan + ' -> UCTAN UCA DOGRULA') if kazanan else 'HICBIRI GECMEDI'}")
    json.dump(sonuc | {"kazanan": kazanan}, open("results/u3_parca_ici.json", "w"), indent=1)
    print("makbuz -> results/u3_parca_ici.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
