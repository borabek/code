# -*- coding: utf-8 -*-
"""GUVEN KALIBRASYONU: "bu isaret dogru mu?" sorusuna AYRI bir model.

OLCULDU (D7, 2512 tahmin): ham skorla kesinlik HICBIR esikte >=0.90 olmuyor,
en yuksek 0.6429 ve egri tepe sonrasi GERI DONUYOR. Sebep, karar kuralinin
GORELI olmasi (`0.85 x parca-maks`): secilen tahminlerin hepsi zaten parca
maksimumuna yakin, dolayisiyla MUTLAK skor parcalar arasi ayirt edici degil.

Bu modul SECILMIS tahminler uzerinde ikinci bir soru sorar: "bu tahmin DOGRU
mu?" Girdi, secim aninda zaten hesaplanmis olan sinyallerdir:

  * ham skor ve parca-ici GORELI konumu (skor / parca-maks, sira yuzdeligi)
  * secenek bankasinin o konumda ne kadar HEMFIKIR oldugu
  * kafes / sira tutarliligi
  * agiz olculeri (girme, erisim, narinlik)
  * parca baglami (aday sayisi, secilen tahmin sayisi)

Cikti kalibre bir olasiliktir; guven kapisi ONUN uzerine kurulur. Boylece
"robotun kullandigi isaretler >=0.90 kesinliktedir" sozu, KAPSAMA bedeliyle
birlikte verilebilir hale gelir.

Egitim `tam`+`d6`, dogrulama MARKA KATLARINDA. D7'ye BAKILMAZ.
"""
import collections
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_u25")
sys.path.insert(0, ".")
import kafes                                 # noqa: E402
import kanonik_d7 as K                       # noqa: E402
import p6_karar                              # noqa: E402
import yon_bankasi as YB                     # noqa: E402
from kos_p6_ortak import yukle               # noqa: E402
from sina_kume import esle_macar             # noqa: E402

PAKET = os.environ.get("P6_MODEL", "results/p6_kademe2_model.pkl")
HEDEF = float(os.environ.get("KAL_HEDEF", "0.90"))
OZ_AD = ["skor", "skor_orani", "skor_sira", "skor_marj",
         "banka_hemfikir", "banka_n", "kafes_mesafe", "kafes_yon",
         "girme", "erisim", "narinlik", "yaricap",
         "n_aday", "n_secilen", "secim_sirasi"]


def secim_ve_oznitelik(d, pk):
    """P6 secimini yap ve HER SECILEN TAHMIN icin kalibrasyon ozniteligi uret."""
    Xd = np.hstack([p6_karar.donustur(d["X"], pk.get("zskor", "ab")),
                    p6_karar.kaynak_blok(d["kaynak"][d["idx"]])])
    if pk.get("kol") == "P6_GEO":
        Xd = np.hstack([Xd[:, 58:p6_karar.AB], Xd[:, p6_karar.AB:]])
    s = np.asarray(pk["kademe1"].predict_proba(
        Xd.astype(np.float32))[:, 1], float)
    P, D, ai, sc = p6_karar.sec_ayrintili(
        d["P"], d["idx"], d["YD"], s, tuple(pk["kural"]),
        nms_mm=float(pk["nms"]))
    if not len(P):
        return P, D, np.zeros((0, len(OZ_AD)))
    smax = float(s.max()) if len(s) else 1.0
    sira = np.argsort(np.argsort(-s)) / max(len(s) - 1, 1)
    Pt, Dt = p6_karar.sec_ayrintili(
        d["P"], d["idx"], d["YD"], s, tuple(pk["tohum_kural"]),
        nms_mm=float(pk["tohum_nms"]))[:2]
    kb = kafes.oznitelik(d["P"][d["idx"]], d["YD"], Pt, Dt)
    D_blok = d["X"][:, p6_karar.AB + len(YB.OZ_AD):]      # agiz olculeri (9)
    X = []
    for r, i in enumerate(ai):
        j = np.where(d["idx"] == i)[0]                     # bu adayin secenegi
        sj = s[j]
        # bankanin hemfikirligi: bu adayin secenekleri ne kadar tek noktada
        hem = float(sj.max() - np.median(sj)) if len(sj) > 1 else 0.0
        # secilen secenegin satiri: skoru sc[r] olan
        t = j[int(np.argmin(np.abs(sj - sc[r])))]
        X.append([sc[r], sc[r] / max(smax, 1e-9), float(sira[t]), hem,
                  float(sj.max() - sj.min()) if len(sj) > 1 else 0.0,
                  float(len(sj)), float(kb[t, 1]), float(kb[t, 6]),
                  float(D_blok[t, 3]), float(D_blok[t, 5]),
                  float(D_blok[t, 2]), float(D_blok[t, 0]),
                  float(len(d["P"])), float(len(P)), float(r)])
    return P, D, np.asarray(X, float)


def dogru_maskesi(P, D, d):
    tp, fp, fn, bilgi = esle_macar(P, D, d["G"], d["Gd"], d["diag"],
                                   K.YANAL, K.ACI, False, isaretli=True)
    e = {int(x[0]) for x in bilgi.get("eslesme", [])}
    return np.array([1 if i in e else 0 for i in range(len(P))], int)


def egri(S, Y):
    i = np.argsort(-S)
    y = Y[i]
    d = np.cumsum(y)
    n = np.arange(1, len(y) + 1)
    return S[i], d / n, n / len(y)


def main():
    t0 = time.time()
    pk = pickle.load(open(PAKET, "rb"))
    veri = []
    for kume in os.environ.get("P6_KUME", "tam,d6").split(","):
        veri += yukle(kume.strip(), int(os.environ.get("P6_TR", "0")))
    print(f"{len(veri)} parca ({time.time() - t0:.0f} s)", flush=True)

    X, Y, M = [], [], []
    for i, d in enumerate(veri, 1):
        P, D, x = secim_ve_oznitelik(d, pk)
        if not len(P):
            continue
        y = dogru_maskesi(P, D, d)
        X.append(x)
        Y.append(y)
        M += [d["mfg"]] * len(y)
        if i % 600 == 0:
            print(f"  {i}/{len(veri)} ({time.time() - t0:.0f} s)", flush=True)
    X = np.vstack(X)
    Y = np.concatenate(Y)
    M = np.asarray(M)
    print(f"tahmin {len(Y)} | ham kesinlik {Y.mean():.4f}", flush=True)

    marka = collections.Counter(M)
    katlar = [m for m, n in marka.items() if n >= 400]
    print(f"katlar: {katlar}", flush=True)

    # HAM SKOR ile kiyas (ilk sutun)
    hs, hk, hc = egri(X[:, 0], Y)
    print(f"\nHAM SKOR: en yuksek kesinlik {hk.max():.4f}")

    Sk = np.zeros(len(Y))
    for b in katlar:
        ic = M != b
        dis = M == b
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0).fit(X[ic], Y[ic])
        Sk[dis] = m.predict_proba(X[dis])[:, 1]
    kd = np.isin(M, katlar)
    ks, kk, kc = egri(Sk[kd], Y[kd])
    print(f"KALIBRE  : en yuksek kesinlik {kk.max():.4f}")

    print(f"\n{'hedef':>7}{'ham kapsama':>14}{'kalibre kapsama':>18}")
    out = {}
    for h in (0.60, 0.70, 0.80, 0.90):
        a = hc[hk >= h]
        b = kc[kk >= h]
        ha = float(a.max()) if len(a) else 0.0
        kb_ = float(b.max()) if len(b) else 0.0
        out[str(h)] = {"ham_kapsama": ha, "kalibre_kapsama": kb_}
        print(f"{h:>7.2f}{ha:>14.4f}{kb_:>18.4f}")

    son = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=0).fit(X, Y)
    pk["kalibrasyon"] = {"model": son, "oz_ad": OZ_AD}
    with open(PAKET, "wb") as f:
        pickle.dump(pk, f)
    json.dump({"damga": makbuz_hash.damga(), "n_tahmin": int(len(Y)),
               "ham_kesinlik": float(Y.mean()),
               "ham_maks_kesinlik": float(hk.max()),
               "kalibre_maks_kesinlik": float(kk.max()),
               "kapsama": out, "katlar": katlar,
               "not": "Guven kalibrasyonu: secilmis tahminler uzerinde ikinci "
                      "model. Kat-disi olcum; D7'ye BAKILMADI."},
              open("results/kalibrasyon.json", "w"), indent=1)
    print(f"\nmakbuz -> results/kalibrasyon.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
