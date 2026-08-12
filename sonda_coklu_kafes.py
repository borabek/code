# -*- coding: utf-8 -*-
"""VII.0 -- COKLU KAFES TAVANI: 0.70 mumkun mu?

TEK OTELEME sondasi NIT icin tekrar tavanini **0.557** olctu. 0.70 hedefi ise
NIT'te 0.66 istiyor, yani tek-oteleme tavaninin (%92'si) neredeyse tamamini
ve HIC yanlis pozitif olmamasini. Gercekci degil.

AMA klemenslerde cogu zaman BIRDEN COK dizi vardir (iki sira, iki kat, on/arka
yuz). Tavan tek otelemeyle sinirli DEGIL. Bu sonda acgozlu olarak birden cok
kafes cikarir ve KUMULATIF kapsamayi olcer:

    1 kafes -> ? | 2 kafes -> ? | 3 kafes -> ?

EGER 3 kafesle NIT 0.80+'a cikiyorsa 0.70 KONUSULABILIR hale gelir.
Cikmiyorsa 0.70 yapisal olarak kapali demektir ve bunu simdi bilmek iyidir.

AYRICA HARMONIK TUZAGI SINANIR: bulunan adim L icin L/2 ve L/3 de denenir.
Onceki sonda NIT'te 10.50mm buldu, oysa standart klemens adimlari 3.5-7.5mm --
yani muhtemelen 2x harmonik bulunuyor ve URETILEN IZGARA HER IKINCI CP'YI
ATLIYOR. Alt harmonik daha cok GT tutuyorsa tavan YUKSELIR.

MODEL EGITIMI YOK -- yalnizca GT geometrisi. Hizli.
D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K       # noqa: E402
import d6_kayit              # noqa: E402

TOL = float(os.environ.get("CK_TOL", "1.0"))     # izgara-GT esleme toleransi
N_KAFES = int(os.environ.get("CK_N", "3"))
KATALOG = (3.5, 3.81, 5.0, 5.08, 6.2, 7.5, 7.62, 10.16, 12.7)


def _izgara_tut(G, tohum, adim):
    """`tohum + n*adim` izgarasina TOL icinde dusen GT maskesi."""
    L = float(np.linalg.norm(adim))
    if L < 1e-6:
        return np.zeros(len(G), bool)
    u = adim / L
    v = G - tohum
    t = v @ u
    dik = np.linalg.norm(v - t[:, None] * u[None, :], axis=1)
    n = np.round(t / L)
    hata = np.abs(t - n * L)
    return (dik <= TOL) & (hata <= TOL)


def _en_iyi_kafes(G, kalan):
    """Kalan GT'leri EN COK kapsayan (tohum, adim) ikilisi."""
    idx = np.where(kalan)[0]
    if len(idx) < 3:
        return None, 0, 0.0
    Gk = G[idx]
    farklar = (Gk[:, None, :] - Gk[None, :, :]).reshape(-1, 3)
    boy = np.linalg.norm(farklar, axis=1)
    iyi = (boy > 0.5) & (boy < np.percentile(boy[boy > 0.5], 50))
    aday = farklar[iyi]
    if not len(aday):
        return None, 0, 0.0
    if len(aday) > 300:
        aday = aday[np.random.default_rng(0).choice(len(aday), 300, False)]
    # ALT HARMONIKLER de aday: L, L/2, L/3
    genis = [aday]
    for b in (2.0, 3.0):
        genis.append(aday / b)
    aday = np.vstack(genis)
    en_maske, en_n, en_L = None, 0, 0.0
    for adim in aday:
        for tohum in Gk[:: max(1, len(Gk) // 8)]:
            m = _izgara_tut(G, tohum, adim) & kalan
            n = int(m.sum())
            if n > en_n:
                en_maske, en_n, en_L = m, n, float(np.linalg.norm(adim))
    return en_maske, en_n, en_L


def coklu(G, n_kafes=N_KAFES):
    """Acgozlu: en iyi kafesi bul, kapsananlari cikar, tekrarla."""
    G = np.asarray(G, float)
    kalan = np.ones(len(G), bool)
    kapsam, adimlar = [], []
    for _ in range(n_kafes):
        m, n, L = _en_iyi_kafes(G, kalan)
        if m is None or n <= 1:
            break
        kalan = kalan & ~m
        kapsam.append(1.0 - kalan.mean())
        adimlar.append(round(L, 2))
    while len(kapsam) < n_kafes:
        kapsam.append(kapsam[-1] if kapsam else 0.0)
    return kapsam, adimlar


def main():
    t0 = time.time()
    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    hedef = set(os.environ.get("CK_MARKA", "NIT,MOR,SUPU,UPUN").split(","))
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for pid, r in kay.items():
        mfg = r.get("mfg")
        if mfg not in hedef:
            continue
        G = np.asarray(r.get("G", []), float)
        if len(G) < 3:
            continue
        kaps, adim = coklu(G)
        a = ist[mfg]
        a["gt"].append(len(G))
        for i, k in enumerate(kaps):
            a[f"kafes{i + 1}"].append(k * len(G))
        if adim:
            a["adim1"].append(adim[0])
        n += 1
        if n % 50 == 0:
            print(f"  {n} parca ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{'marka':<7}{'parca':>6}{'GT':>7}{'1 kafes':>9}{'2 kafes':>9}"
          f"{'3 kafes':>9}{'adim1':>8}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        g = sum(a["gt"])
        r = {"parca": len(a["gt"]), "gt": g}
        for i in (1, 2, 3):
            r[f"kafes{i}"] = sum(a[f"kafes{i}"]) / max(g, 1)
        r["adim1_ortanca"] = float(np.median(a["adim1"])) if a["adim1"] else 0.0
        out[m_] = r
        print(f"{m_:<7}{r['parca']:>6}{g:>7}{r['kafes1']:>9.3f}"
              f"{r['kafes2']:>9.3f}{r['kafes3']:>9.3f}"
              f"{r['adim1_ortanca']:>8.2f}")

    # 0.70 KARARI
    nit = out.get("NIT", {})
    k3 = nit.get("kafes3", 0.0)
    f1_tav = 2 * k3 / (1 + k3) if k3 else 0.0
    print(f"\n0.70 KARARI -- NIT 3 kafeste kapsama {k3:.3f} "
          f"-> F1 tavani {f1_tav:.4f}")
    print(f"  0.70 icin NIT'te ~0.66 gerekiyordu.")
    print(f"  {'KONUSULABILIR' if f1_tav >= 0.75 else 'YAPISAL OLARAK ZOR'}"
          f" (tavan {f1_tav:.3f})")
    json.dump({"tol": TOL, "n_kafes": N_KAFES, "marka": out,
               "nit_3kafes_f1_tavani": f1_tav,
               "not": "Acgozlu coklu kafes: en iyi izgarayi bul, kapsananlari "
                      "cikar, tekrarla. Alt harmonikler (L/2, L/3) de aday. "
                      "MODEL YOK -- yalnizca GT geometrisi. D7'ye BAKILMADI."},
              open("results/coklu_kafes.json", "w"), indent=1)
    print("makbuz -> results/coklu_kafes.json")


if __name__ == "__main__":
    main()
