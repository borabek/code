# -*- coding: utf-8 -*-
"""VII.0c-g -- KAFES ARAMASI, GT'SIZ OLCUTLE (uretimde yapilabilir haliyle)

VII.0b'DEKI KUSUR. Kafesi ararken hedef fonksiyon olarak **GT kapsamasini**
kullanmistim. Yani arama KAHIN tarafindan yonlendiriliyordu; uretimde boyle
bir sey yapilamaz. O yuzden NIT'in 0.323'u "arama zayif" degil, daha kotusu:
kahin yardimiyla bile skorla filtrelenmis aday bulutundan iyi kafes cikmiyor.

BU SURUMDE UC SEY DEGISTI:
  1. ARAMA OLCUTU GT'SIZ: kafes, uzerine dusen ADAY sayisi ve izgara
     DOLULUGU ile puanlanir. GT yalnizca DEGERLENDIRMEDE kullanilir.
  2. TAM HAVUZ: adaylar skora gore filtrelenmez. Kafes GEOMETRIK bir
     ozelliktir; NIT'te GT'nin yalnizca %5.3'u ilk-k icinde oldugu icin skor
     filtresi kafesi YOK EDIYORDU.
  3. ADIL TOLERANS + DOYGUNLUK: kahinle ayni tolerans, ve 6 kafese kadar
     (NIT'te kapsama 3'te hala tirmaniyordu: 0.161/0.255/0.323).

Model EGITIMI YOK -- yalnizca aday geometrisi ve GT (degerlendirme icin).
D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam4")
sys.path.insert(0, ".")
from kos_p6_ortak import yukle     # noqa: E402

KUME = os.environ.get("KV_KUME", "d6")
TOL = float(os.environ.get("KV_TOL", "1.0"))       # kahinle AYNI
N_KAFES = int(os.environ.get("KV_N", "6"))
MIN_ADIM = float(os.environ.get("KV_MIN_ADIM", "2.0"))
MARKALAR = set(os.environ.get("KV_MARKA", "NIT,MOR,SUPU,UPUN").split(","))


def _izgara(P, tohum, adim, n_max=80):
    L = float(np.linalg.norm(adim))
    if L < MIN_ADIM:
        return np.zeros((0, 3)), 0
    u = adim / L
    t = (P - tohum) @ u
    n0 = max(int(np.floor(t.min() / L)) - 1, -n_max)
    n1 = min(int(np.ceil(t.max() / L)) + 1, n_max)
    ns = np.arange(n0, n1 + 1)
    return tohum[None, :] + ns[:, None] * (u * L)[None, :], len(ns)


def _yakin(A, B, tol):
    """A'nin her satiri B'ye tol icinde mi."""
    if not len(A) or not len(B):
        return np.zeros(len(A), bool)
    return (np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1).min(1)
            <= tol)


def kafes_ara(P, n_kafes=N_KAFES, rng=None):
    """GT'SIZ arama: izgarayi 'uzerine dusen ADAY sayisi x DOLULUK' ile puanla.

    DOLULUK = isabet / izgara noktasi. Bu olmadan arama COK KUCUK adimi secer
    (her seyi kapsar ama anlamsizdir); MIN_ADIM ile birlikte iki yonlu koruma.
    """
    rng = rng or np.random.default_rng(0)
    kalan_aday = np.ones(len(P), bool)
    bulunan = []
    if len(P) < 3:
        return bulunan
    for _ in range(n_kafes):
        idx = np.where(kalan_aday)[0]
        if len(idx) < 3:
            break
        Pk = P[idx]
        farklar = (Pk[:, None, :] - Pk[None, :, :]).reshape(-1, 3)
        boy = np.linalg.norm(farklar, axis=1)
        iyi = boy >= MIN_ADIM
        if not iyi.any():
            break
        aday = farklar[iyi & (boy <= np.percentile(boy[iyi], 60))]
        if not len(aday):
            break
        if len(aday) > 200:
            aday = aday[rng.choice(len(aday), 200, replace=False)]
        aday = np.vstack([aday, aday / 2.0, aday / 3.0])      # alt harmonikler
        tohumlar = Pk[:: max(1, len(Pk) // 12)]
        en = (None, -1.0, None)
        for adim in aday:
            if np.linalg.norm(adim) < MIN_ADIM:
                continue
            for tohum in tohumlar:
                uret, n_gr = _izgara(P, tohum, adim)
                if n_gr < 3:
                    continue
                isabet = _yakin(uret, P, TOL)
                n_hit = int(isabet.sum())
                if n_hit < 3:
                    continue
                doluluk = n_hit / max(n_gr, 1)
                puan = n_hit * doluluk          # GT KULLANILMIYOR
                if puan > en[1]:
                    en = ((tohum, adim), puan, uret)
        if en[0] is None:
            break
        bulunan.append(en)
        # bu kafesin uzerine dusen ADAYLARI cikar
        kapsanan = _yakin(P, en[2], TOL)
        kalan_aday = kalan_aday & ~kapsanan
    return bulunan


def main():
    t0 = time.time()
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for d in veri:
        if d["mfg"] not in MARKALAR:
            continue
        G = np.asarray(d["G"], float)
        if len(G) < 3:
            continue
        P = np.unique(np.round(np.asarray(d["P"], float), 3), axis=0)
        bulunan = kafes_ara(P)
        a = ist[d["mfg"]]
        a["gt"].append(len(G))
        a["aday"].append(len(P))
        kalan = np.ones(len(G), bool)
        for i in range(N_KAFES):
            if i < len(bulunan):
                kalan = kalan & ~_yakin(G, bulunan[i][2], TOL)
            a[f"k{i + 1}"].append(int((~kalan).sum()))
        if bulunan:
            a["adim1"].append(float(np.linalg.norm(bulunan[0][0][1])))
        n += 1
        if n % 40 == 0:
            print(f"  {n} parca ({time.time() - t0:.0f} s)", flush=True)

    kahin = {"NIT": 0.983, "SUPU": 0.864, "MOR": 0.811, "UPUN": 0.810}
    bas = "".join(f"{i}k".rjust(7) for i in range(1, N_KAFES + 1))
    print(f"\n{'marka':<7}{'GT':>6}{'aday':>6}{bas}{'KAHIN':>8}{'acik':>7}"
          f"{'adim1':>8}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        g = sum(a["gt"])
        r = {"gt": g, "aday_parca": float(np.mean(a["aday"]))}
        sat = ""
        for i in range(1, N_KAFES + 1):
            v = sum(a[f"k{i}"]) / max(g, 1)
            r[f"kafes{i}"] = v
            sat += f"{v:>7.3f}"
        kh = kahin.get(m_, 0.0)
        r["kahin"] = kh
        r["acik"] = kh - r[f"kafes{N_KAFES}"]
        r["adim1_ortanca"] = float(np.median(a["adim1"])) if a["adim1"] else 0.0
        out[m_] = r
        print(f"{m_:<7}{g:>6}{r['aday_parca']:>6.0f}{sat}{kh:>8.3f}"
              f"{r['acik']:>7.3f}{r['adim1_ortanca']:>8.2f}")

    json.dump({"tol": TOL, "n_kafes": N_KAFES, "min_adim": MIN_ADIM,
               "marka": out,
               "not": "ARAMA GT'SIZ: izgara 'isabet x doluluk' ile puanlanir, "
                      "GT yalniz DEGERLENDIRMEDE. Adaylar skorla "
                      "FILTRELENMEZ (kafes geometrik bir ozelliktir). "
                      "D7'ye BAKILMADI."},
              open(f"results/kafes_v2_{KUME}.json", "w"), indent=1)
    print(f"\nmakbuz -> results/kafes_v2_{KUME}.json")


if __name__ == "__main__":
    main()
