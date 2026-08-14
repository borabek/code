# -*- coding: utf-8 -*-
"""POSE KIRPMA SINIRI (`maks_mm`) TARAMASI -- cevrimdisi, tek kosudan.

GEREKCE (2026-08-14 teshisi, Bolum 21.46): baglayici kisit YANAL KONUM.
GT'lerin **%83'u** icin havuzda, yonu zaten 10 derece icinde dogru olan
bir aday **10 mm** yakinda duruyor; ama pose head'in duzeltmesi
**3 mm**'ye kirpiliyor (`pose_head.pkl: maks_mm=3.0`). Bu hiperparametrenin
tarandigina dair kayit YOK.

Sonda, kirpma ONCESI yer degistirme vektorunu dokuyor (`pose_dw`), bu
yuzden TEK kosudan her `maks_mm` degeri cevrimdisi yeniden kurulabilir.

DOGRULAMA: mx = 3.0'da yeniden kurulan metrik, dokumun kendi metrigine
ESIT olmali. Esit degilse tarama gecersizdir ve betik bunu SOYLER.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sina_kume import esle_macar                       # noqa: E402

DOKUM = os.environ.get("PK_DOKUM", "results/_dokum_pose.json")
YOL = os.environ.get("PK_YOL", "saha")
MEVCUT = 3.0


def _birim(v):
    v = np.asarray(v, float).reshape(-1, 3)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def kur(r, mx, guven_esik=None, mx_dusuk=MEVCUT):
    """kirpma siniri mx ile noktalari yeniden kur.

    `guven_esik` verilirse KOSULLU kirpma: guveni esigin ustunde olan
    adaylarda sinir `mx`, digerlerinde `mx_dusuk` (mevcut 3 mm). Gerekce:
    buyuk duzeltmeye ancak modelin emin oldugu yerde izin vermek, "emin
    degilse dokunma" ilkesinin bu koldaki karsiligidir.
    """
    p0 = np.asarray(r.get("pose_p0") or [], float).reshape(-1, 3)
    dw = np.asarray(r.get("pose_dw") or [], float).reshape(-1, 3)
    P = np.asarray(r["P"], float).reshape(-1, 3)
    D = _birim(r["D"]) if len(P) else np.zeros((0, 3))
    if not len(p0) or len(p0) != len(P):
        # pose kancasi ile cikti hizalanmiyor (pose sonrasi eleme olmus)
        return None, None
    n = np.linalg.norm(dw, axis=1)
    if guven_esik is None:
        sinir = np.full(len(n), float(mx))
    else:
        g = np.asarray(r.get("guven") or [], float).reshape(-1)
        if len(g) != len(n):
            return None, None
        g = np.where(np.isfinite(g), g, -np.inf)
        sinir = np.where(g >= guven_esik, float(mx), float(mx_dusuk))
    ol = np.where(n > 1e-9, np.minimum(n, sinir) / np.maximum(n, 1e-12), 0.0)
    return p0 + dw * ol[:, None], D


def olc(kayit, mx=None, guven_esik=None):
    tot = {k: [0, 0, 0] for k in ("tespit", "rob", "rbi")}
    parca, atlanan = [], 0
    for r in kayit:
        G = np.asarray(r["G"], float).reshape(-1, 3)
        if not len(G):
            continue
        Gd = _birim(r["Gd"])
        if mx is None:
            P = np.asarray(r["P"], float).reshape(-1, 3)
            D = _birim(r["D"]) if len(P) else np.zeros((0, 3))
        else:
            P, D = kur(r, mx, guven_esik)
            if P is None:
                atlanan += 1
                P = np.asarray(r["P"], float).reshape(-1, 3)
                D = _birim(r["D"]) if len(P) else np.zeros((0, 3))
        satir = {}
        for ad, (tol, am, isr) in (("tespit", (2.0, 180.0, False)),
                                   ("rob", (2.0, 10.0, False)),
                                   ("rbi", (2.0, 10.0, True))):
            tp, fp, fn, _ = esle_macar(P, D, G, Gd, float(r["diag"]),
                                       tol, am, False, isaretli=isr)
            tot[ad][0] += tp
            tot[ad][1] += fp
            tot[ad][2] += fn
            satir[ad] = (tp, fp, fn)
        parca.append(satir)
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)   # noqa: E731
    return {k: f1(v) for k, v in tot.items()}, parca, atlanan


def boot(pa, pb, ad, n=4000, tohum=0):
    rng = np.random.default_rng(tohum)
    A = np.asarray([r[ad] for r in pa], float)
    B = np.asarray([r[ad] for r in pb], float)
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)   # noqa: E731
    d = [f1(B[i].sum(0)) - f1(A[i].sum(0))
         for i in (rng.integers(0, len(A), len(A)) for _ in range(n))]
    d = np.asarray(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


def main():
    kayit = [r for r in json.load(open(DOKUM)) if r.get("yol") == YOL]
    print(f"{DOKUM} / yol={YOL} -> {len(kayit)} parca")
    kanca_var = sum(1 for r in kayit if r.get("pose_dw"))
    print(f"pose kancasi olan parca: {kanca_var}")
    if not kanca_var:
        print("POSE KANCASI BOS -- tarama yapilamaz")
        return 1

    dok, dok_p, _ = olc(kayit, None)
    kur3, kur3_p, atl = olc(kayit, MEVCUT)
    print(f"\nDOGRULAMA (mx={MEVCUT} yeniden kurulan == dokum?)  "
          f"hizalanmayan parca: {atl}")
    print(f"  dokum       : tespit {dok['tespit']:.4f} rob {dok['rob']:.4f} "
          f"rbi {dok['rbi']:.4f}")
    print(f"  yeniden kur : tespit {kur3['tespit']:.4f} "
          f"rob {kur3['rob']:.4f} rbi {kur3['rbi']:.4f}")
    ok = all(abs(dok[k] - kur3[k]) < 1e-6 for k in dok)
    print(f"  -> {'GECERLI' if ok else 'UYUSMUYOR -- tarama SUPHELI'}")

    MX = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 1e9]
    sonuc = {}
    print(f"\n{'maks_mm':>9s} {'tespit':>8s} {'rob':>8s} {'rob-ISR':>8s}")
    for mx in MX:
        m, p, _ = olc(kayit, mx)
        sonuc[mx] = (m, p)
        ad = "SINIRSIZ" if mx > 1e8 else f"{mx:.1f}"
        yz = "  <- MEVCUT" if mx == MEVCUT else ""
        print(f"{ad:>9s} {m['tespit']:8.4f} {m['rob']:8.4f} "
              f"{m['rbi']:8.4f}{yz}")

    print("\n--- ESLI BOOTSTRAP (mevcut 3.0'a gore) ---")
    print(f"{'maks_mm':>9s} {'metrik':>7s} {'fark':>9s} {'%95 GA':>22s} "
          f"{'poz%':>6s}")
    taban = sonuc[MEVCUT][1]
    for mx in MX:
        if mx == MEVCUT:
            continue
        for ad in ("rob", "rbi"):
            f, lo, hi, pz = boot(taban, sonuc[mx][1], ad)
            yz = " *" if (lo > 0 or hi < 0) else ""
            nm = "SINIRSIZ" if mx > 1e8 else f"{mx:.1f}"
            print(f"{nm:>9s} {ad:>7s} {f:+9.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]{yz:>3s} {100*pz:5.1f}")

    # GUVEN KAPILI KIRPMA: buyuk duzeltmeye yalniz emin oldugu yerde izin
    en_iyi = max((k for k in MX if k <= 1e8),
                 key=lambda k: sonuc[k][0]["rbi"])
    if en_iyi != MEVCUT:
        print(f"\n--- GUVEN KAPILI KIRPMA (emin ise {en_iyi:.1f} mm, "
              f"degilse {MEVCUT:.1f} mm) ---")
        print(f"{'esik':>6s} {'tespit':>8s} {'rob':>8s} {'rob-ISR':>8s} "
              f"{'fark(rbi)':>10s} {'%95 GA':>22s}")
        for esik in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            m, p, _ = olc(kayit, en_iyi, esik)
            f, lo, hi, _ = boot(taban, p, "rbi")
            yz = " *" if (lo > 0 or hi < 0) else ""
            print(f"{esik:6.2f} {m['tespit']:8.4f} {m['rob']:8.4f} "
                  f"{m['rbi']:8.4f} {f:+10.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]{yz:>3s}")

    # yer degistirme buyuklugu dagilimi -- kirpmanin ne kadar bagladigi
    n = np.concatenate([np.linalg.norm(np.asarray(r["pose_dw"], float)
                                       .reshape(-1, 3), axis=1)
                        for r in kayit if r.get("pose_dw")])
    print(f"\nOnerilen yer degistirme (|dw|, {len(n)} CP): "
          f"ortanca {np.median(n):.2f} mm, %75 {np.percentile(n, 75):.2f}, "
          f"%90 {np.percentile(n, 90):.2f}, maks {n.max():.2f}")
    print(f"3 mm'yi ASAN oneri orani: {100*(n > 3).mean():.1f}%")

    json.dump({("SINIRSIZ" if k > 1e8 else k): v[0]
               for k, v in sonuc.items()},
              open("results/pose_kirpma.json", "w"), indent=1)
    print("\n-> results/pose_kirpma.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
