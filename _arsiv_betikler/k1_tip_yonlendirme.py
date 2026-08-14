# -*- coding: utf-8 -*-
"""K1: ADAY TIPINE GORE YONLENDIRME -- kendi listemin birinci kolu.

GOZLEM (bu gece T4'ten): bosluk-grafi ozellikleri adaylarin yalniz %52.4'unde
hesaplanabiliyor. O yari, eseksenli bir B-rep SILINDIRI olanlar = YUVARLAK KANAL.
Diger yari kare/kelepce/yarik girisler. Bunlar FIZIKSEL OLARAK FARKLI nesneler ve
tek bir gate ikisini AYNI ANDA ogrenmeye calisiyor.

NEDEN ONCEKI YONLENDIRMENIN TEKRARI DEGIL: mevcut router PARCA duzeyinde ve rejime
(dusuk-CP / cok-CP) gore calisiyor. Bu ADAY duzeyinde ve GEOMETRIK TIPE gore. Farkli eksen.
Ayrica tip belirteci DETERMINISTIK ve calisma aninda mevcut (B-rep'ten gelir, agdan degil).

HIPOTEZ: iki alt-popülasyonun ayirt edici ozellikleri farklidir; ayri gate'ler her birinde
daha keskin olur. Hafizadaki [[two-regimes-low-vs-high-cp]] ayni dersi parca duzeyinde
verdi (duz ortalama YANILTIR).

KILL (onceden yazildi): aday duzeyi +0.01 gelmezse kol kapanir. Gecerse uctan uca sinav.
"""
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import gate_tezgah as T
    import olcum_kumesi
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from gece_kilit import bekci

    bekci("k1")
    d4 = np.load("results/t4_bosluk.npz", allow_pickle=True)
    BG = d4["BG"]; RY = d4["RY"]
    RG = np.array([str(x) for x in d4["RG"]]); RP = np.array([str(x) for x in d4["RP"]])
    D = T.yukle()
    olcum_kumesi.rapor_bas(D["rap"])
    RX = []
    for r in D["DER"]:
        if r["X"] is None or r.get("XR") is None or not len(r["G"]):
            continue
        RX.append(np.hstack([r["X"], r["XR"]]))
    RX = np.vstack(RX)
    assert len(RX) == len(BG) == len(RY)

    # TIP: eseksenli B-rep silindiri VAR mi (deterministik, calisma aninda mevcut)
    tip = (BG[:, 0] > 0)
    print(f"\n{len(RY)} aday | YUVARLAK KANAL {tip.mean():.1%} | DIGER {(~tip).mean():.1%}")
    print(f"  pozitif orani: yuvarlak {RY[tip].mean():.1%} | diger {RY[~tip].mean():.1%}")

    def karar(o):
        m = np.zeros(len(o), bool)
        for u in np.unique(RP):
            i = RP == u; v = o[i]
            m[i] = (v >= 0.5 * max(v.max(), 1e-9)) & (v >= 0.25)
        return m

    def f1(m):
        tp = int((RY.astype(bool) & m).sum()); fp = int((~RY.astype(bool) & m).sum())
        fn = int((RY.astype(bool) & ~m).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-9)

    def tek_gate(M):
        o = np.zeros(len(RY))
        for tr, te in GroupKFold(n_splits=5).split(M, RY, RG):
            o[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                           random_state=0).fit(M[tr], RY[tr]).predict_proba(M[te])[:, 1]
        return o

    def tipli_gate(M):
        """Her TIP icin AYRI gate. Bolme yine GRUP-CAPRAZ (sizinti yok)."""
        o = np.zeros(len(RY))
        for tr, te in GroupKFold(n_splits=5).split(M, RY, RG):
            for t in (True, False):
                tr_t = tr[tip[tr] == t]; te_t = te[tip[te] == t]
                if len(te_t) == 0:
                    continue
                if len(tr_t) < 50 or len(np.unique(RY[tr_t])) < 2:
                    tr_t = tr                       # yeterli ornek yoksa tum egitimi kullan
                o[te_t] = RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                    random_state=0).fit(M[tr_t], RY[tr_t]).predict_proba(M[te_t])[:, 1]
        return o

    M58 = RX
    M66 = np.hstack([RX, BG])
    SON = {}
    for ad, M, fn_ in (("58 tek gate", M58, tek_gate),
                       ("58 TIPLI gate", M58, tipli_gate),
                       ("58+bg tek gate", M66, tek_gate),
                       ("58+bg TIPLI gate", M66, tipli_gate)):
        o = fn_(M); SON[ad] = f1(karar(o))
        print(f"{ad:<22}{SON[ad]:.4f}")
    taban = SON["58 tek gate"]
    en_iyi = max((v, k) for k, v in SON.items() if k != "58 tek gate")
    print(f"\ntaban {taban:.4f} -> en iyi {en_iyi[1]} {en_iyi[0]:.4f} ({en_iyi[0]-taban:+.4f})")
    gecti = (en_iyi[0] - taban) >= 0.01
    print(f"KARAR: {'SINYAL VAR -> uctan uca sinava' if gecti else 'SINYAL YOK -> K1 KAPANIR'}")

    # TIP BASINA AYRISTIRMA (nerede kazaniyor/kaybediyor)
    o1 = tek_gate(M58); o2 = tipli_gate(M58)
    for t, ad in ((True, "yuvarlak kanal"), (False, "diger")):
        m1 = karar(o1); m2 = karar(o2)
        def f1_alt(m, msk):
            tp = int((RY.astype(bool) & m & msk).sum()); fp = int((~RY.astype(bool) & m & msk).sum())
            fn = int((RY.astype(bool) & ~m & msk).sum())
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            return 2 * p * r / max(p + r, 1e-9)
        print(f"  {ad:<16} tek {f1_alt(m1, tip == t):.4f} -> tipli {f1_alt(m2, tip == t):.4f}")
    with io.open("results/k1_tip.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in SON.items()} |
                  {"yuvarlak_pay": float(tip.mean()), "gecti": bool(gecti)}, f, indent=1)
    print("makbuz -> results/k1_tip.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
