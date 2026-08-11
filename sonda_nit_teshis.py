# -*- coding: utf-8 -*-
"""NIT COKUSU: siralama mi, kalibrasyon mi, temsil mi?

DURUM. NIT 50 parca / 1222 GT (24.4 CP/parca -- D6'nin en yogun markasi).
Dagitilan urun 1222 GT'den **2** tanesini buluyor (F1 0.0032). Yeni havuz o
markada yonlu recall **0.5254** veriyor, yani cevabin yarisi HAVUZDA. Kural
kahini (o markadaki EN IYI esik/NMS) yalnizca 0.0349 -- demek ki kayip ESIKTE
degil, modelin SIRALAMASINDA.

Bu betik uc soruyu ayirir:

  1. SIRALAMA NE KADAR KOTU?  Esikten bagimsiz olcut: ortalama kesinlik (AP) ve
     `recall@k` (parca basina GT sayisi kadar secenek al). Rastgele siralamanin
     beklenen degeriyle kiyaslanir.
  2. PARCA-ICI Z-SKOR MU BOZUYOR?  Egitim markalarinda parca basina ~150 aday
     var, NIT'te 407. Z-skor her parcayi kendi ortalamasina gore kaydiriyor;
     aday sayisi ve dagilimi cok farkli olunca NIT egitimin HIC GORMEDIGI bir
     bolgeye dusebilir. `zskor=yok` (ham oznitelik) kolu bunu sinar.
  3. HANGI OZNITELIK BLOGU?  A (segmentasyon, 58) / B+D (agiz olculeri, 18) /
     C (yon bankasi, 16) bloklari tek tek verilerek hangisinin NIT'te bilgi
     tasidigi olculur.

Egitim: NIT DISI D6 markalari. Sinav: NIT. (Bu bir TESHIS betigidir; buradan
dagitim karari CIKMAZ, kol secimi `tam` korpusunun katlarinda yapilir.)
"""
import collections
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_u25")
sys.path.insert(0, ".")
import p6_karar                    # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402

AB = p6_karar.AB
A_SUT = 58
HEDEF = os.environ.get("NIT_MARKA", "NIT")


def yap():
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, random_state=0)


def alt(M, Y, kat=6):
    rng = np.random.default_rng(0)
    p = np.where(Y == 1)[0]
    n = np.where(Y == 0)[0]
    k = min(len(n), kat * max(len(p), 1))
    s = np.concatenate([p, rng.choice(n, k, replace=False)])
    rng.shuffle(s)
    return M[s], Y[s]


def blok(d, ad, zskor):
    X = np.hstack([p6_karar.donustur(d["X"], zskor),
                   p6_karar.kaynak_blok(d["kaynak"][d["idx"]])])
    if ad == "hepsi":
        return X
    if ad == "A (segmentasyon)":
        return np.hstack([X[:, :A_SUT], X[:, AB + 16:]])   # A + kaynak
    if ad == "B+D (agiz olculeri)":
        return np.hstack([X[:, A_SUT:AB], X[:, AB + 16:]])
    if ad == "C (yon bankasi)":
        return X[:, AB:]
    raise ValueError(ad)


def ap_ve_recall_k(veri, skor):
    """Esikten BAGIMSIZ siralama olcutleri. Doner: (AP, recall@k, rastgele)."""
    aps, rk, rnd = [], [], []
    for d, s in zip(veri, skor):
        y = np.asarray(d["y"], int)
        if not y.any():
            continue
        sira = np.argsort(-np.asarray(s, float))
        ys = y[sira]
        kum = np.cumsum(ys)
        kes = kum / np.arange(1, len(ys) + 1)
        aps.append(float((kes * ys).sum() / max(ys.sum(), 1)))
        k = int(len(d["G"]))
        rk.append(float(ys[:k].sum()) / max(k, 1))
        rnd.append(float(y.mean()))          # rastgele siralamanin beklentisi
    return (float(np.mean(aps)), float(np.mean(rk)), float(np.mean(rnd)))


def main():
    dev = yukle("d6")
    for d in dev:
        d["y"] = np.asarray(d["y"], int)
    tr = [d for d in dev if d["mfg"] != HEDEF]
    te = [d for d in dev if d["mfg"] == HEDEF]
    if not te:
        sys.exit(f"{HEDEF} yok")
    print(f"egitim {len(tr)} parca ({len(set(d['mfg'] for d in tr))} marka) | "
          f"sinav {HEDEF} {len(te)} parca / "
          f"{sum(len(d['G']) for d in te)} GT", flush=True)
    print(f"aday/parca: egitim {np.mean([len(d['P']) for d in tr]):.0f} | "
          f"{HEDEF} {np.mean([len(d['P']) for d in te]):.0f}")
    print(f"secenek/parca: egitim {np.mean([len(d['idx']) for d in tr]):.0f} | "
          f"{HEDEF} {np.mean([len(d['idx']) for d in te]):.0f}\n")

    out = {}
    print(f"{'kurulum':<34}{'AP':>8}{'recall@k':>10}{'rastgele':>10}{'kat':>7}")
    for zskor in ("ab", "sira", "ikisi", "yok"):
        for ad in ("hepsi", "C (yon bankasi)"):
            M = np.vstack([blok(d, ad, zskor) for d in tr]).astype(np.float32)
            Y = np.concatenate([d["y"] for d in tr])
            M, Y = alt(M, Y)
            m = yap().fit(M, Y)
            sk = [m.predict_proba(blok(d, ad, zskor).astype(np.float32))[:, 1]
                  for d in te]
            ap, rk, rnd = ap_ve_recall_k(te, sk)
            etiket = f"{ad} / zskor={zskor}"
            out[etiket] = {"AP": ap, "recall@k": rk, "rastgele": rnd,
                           "kat": rk / max(rnd, 1e-9)}
            print(f"{etiket:<34}{ap:>8.4f}{rk:>10.4f}{rnd:>10.4f}"
                  f"{rk / max(rnd, 1e-9):>7.1f}x", flush=True)

    json.dump({"damga": makbuz_hash.damga(), "marka": HEDEF, "sonuc": out,
               "not": "TESHIS. Esikten bagimsiz siralama olcutleri. `kat` = "
                      "recall@k'nin rastgele siralamaya orani; 1.0x rastgele "
                      "demektir."},
              open(f"results/nit_teshis_{HEDEF}.json", "w"), indent=1)
    print(f"\nmakbuz -> results/nit_teshis_{HEDEF}.json")
    print("YORUM: `kat` 1.0x civarindaysa model o markada RASTGELE siralama "
          "yapiyor demektir (temsil sorunu). Yuksekse siralama iyi, kayip "
          "esik/kalibrasyonda.")


if __name__ == "__main__":
    main()
