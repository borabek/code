# -*- coding: utf-8 -*-
"""HALKA NORMALI = EKSEN mi? (cevrimdisi, yeni cikarim YOK)

Simdiye kadar halka normali yalnizca **ISARET** duzeltmek icin kullanildi
(`halka_disari`, +0.0195 tanidik markada, KESIN). Ama fiziksel olarak
duz bir yuzeydeki kablo girisinin EKSENI, o yuzeyin normalidir. Yani
halka normali sadece isareti degil, **ekseni de** verebilir.

Bu, isaret kolundan FARKLI bir iddiadir ve simdiye kadar OLCULMEDI:
isaret kollari `isaretsiz` metrigi hic degistirmez (tanim geregi), eksen
degistirmek ise **hem isaretsiz hem isaretli** metrigi degistirir.

UC VARYANT:
  * `tam`      : yonu tamamen halka normaliyle degistir
  * `kapili_A` : yalniz tahmin ile halka normali arasindaki aci A
                 dereceden BUYUKSE degistir (tahmine guvenmedigimiz yer)
  * `harman_w` : w*halka + (1-w)*tahmin, sonra normalize

Karar ESLI PARCA BOOTSTRAP ile verilir.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sina_kume import esle_macar                       # noqa: E402

DOKUM = os.environ.get("HE_DOKUM", "results/_dokum_halka.json")
YOL = os.environ.get("HE_YOL", "saha")


def _birim(v):
    v = np.asarray(v, float).reshape(-1, 3)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def uygula(D, H, kol):
    """D: tahmin yonleri (n,3). H: halka normalleri (n,3), sifir = yok."""
    if not len(D):
        return D
    gecerli = np.linalg.norm(H, axis=1) > 1e-9
    Y = D.copy()
    if kol == "taban":
        return Y
    if kol == "isaret":                       # mevcut, dagitima aday kural
        c = np.sum(D * H, axis=1)
        cevir = gecerli & (c < 0)
        Y[cevir] = -D[cevir]
        return Y
    if kol == "tam":
        Y[gecerli] = H[gecerli]
        return Y
    if kol.startswith("kapili_"):
        A = float(kol.split("_")[1])
        # once isareti hizala, sonra KALAN aci farkina bak
        c = np.sum(D * H, axis=1)
        Hs = H * np.where(c < 0, -1.0, 1.0)[:, None]   # H'yi D'ye yaklastir
        aci = np.degrees(np.arccos(np.clip(np.sum(D * Hs, axis=1), -1, 1)))
        deg = gecerli & (aci > A)
        Y[deg] = H[deg]                        # ISARETIYLE birlikte halka
        # kalanlarda yalnizca isaret duzeltmesi
        kal = gecerli & ~deg & (c < 0)
        Y[kal] = -D[kal]
        return Y
    if kol.startswith("harman_"):
        w = float(kol.split("_")[1])
        c = np.sum(D * H, axis=1)
        Ds = D * np.where(c < 0, -1.0, 1.0)[:, None]   # D'yi H isaretine al
        v = w * H + (1 - w) * Ds
        n = np.linalg.norm(v, axis=1)
        ok = gecerli & (n > 1e-9)
        Y[ok] = v[ok] / n[ok][:, None]
        return Y
    raise ValueError(kol)


def olc(kayit, kol):
    tot = {k: [0, 0, 0] for k in ("tespit", "rob", "rbi")}
    parca = []
    for r in kayit:
        G = np.asarray(r["G"], float).reshape(-1, 3)
        if not len(G):
            continue
        Gd = _birim(r["Gd"])
        P = np.asarray(r["P"], float).reshape(-1, 3)
        D = _birim(r["D"]) if len(P) else np.zeros((0, 3))
        H = np.asarray(r.get("halka_normal") or [], float).reshape(-1, 3)
        if len(H) != len(D):
            H = np.zeros_like(D)
        D2 = uygula(D, _birim(H) if len(H) else H, kol)
        satir = {}
        for ad, (tol, am, isr) in (("tespit", (2.0, 180.0, False)),
                                   ("rob", (2.0, 10.0, False)),
                                   ("rbi", (2.0, 10.0, True))):
            tp, fp, fn, _ = esle_macar(P, D2, G, Gd, float(r["diag"]),
                                       tol, am, False, isaretli=isr)
            tot[ad][0] += tp
            tot[ad][1] += fp
            tot[ad][2] += fn
            satir[ad] = (tp, fp, fn)
        parca.append(satir)
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)   # noqa: E731
    return {k: f1(v) for k, v in tot.items()}, parca


def boot(pa, pb, ad, n=4000, tohum=0):
    rng = np.random.default_rng(tohum)
    A = np.asarray([r[ad] for r in pa], float)
    B = np.asarray([r[ad] for r in pb], float)
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)   # noqa: E731
    d = []
    for _ in range(n):
        i = rng.integers(0, len(A), len(A))
        d.append(f1(B[i].sum(0)) - f1(A[i].sum(0)))
    d = np.asarray(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


def main():
    kayit = [r for r in json.load(open(DOKUM)) if r.get("yol") == YOL
             and r.get("halka_normal")]
    print(f"{DOKUM} / yol={YOL} -> {len(kayit)} parca (halka normali olan)")
    if not kayit:
        return 1
    KOLLAR = ["taban", "isaret", "tam",
              "kapili_10", "kapili_20", "kapili_30", "kapili_45",
              "harman_0.25", "harman_0.5", "harman_0.75"]
    sonuc = {}
    print(f"\n{'kol':14s} {'tespit':>8s} {'rob':>8s} {'rob-ISR':>8s}")
    for kol in KOLLAR:
        m, p = olc(kayit, kol)
        sonuc[kol] = (m, p)
        print(f"{kol:14s} {m['tespit']:8.4f} {m['rob']:8.4f} {m['rbi']:8.4f}")
    print("\n--- ESLI BOOTSTRAP (tabana gore) ---")
    print(f"{'kol':14s} {'metrik':>7s} {'fark':>9s} {'%95 GA':>22s} {'poz%':>6s}")
    for kol in KOLLAR[1:]:
        for ad in ("rob", "rbi"):
            f, lo, hi, pz = boot(sonuc["taban"][1], sonuc[kol][1], ad)
            yz = " *" if (lo > 0 or hi < 0) else ""
            print(f"{kol:14s} {ad:>7s} {f:+9.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]{yz:>3s} {100*pz:5.1f}")
    json.dump({k: v[0] for k, v in sonuc.items()},
              open(f"results/halka_eksen_{YOL}.json", "w"), indent=1)
    print(f"\n-> results/halka_eksen_{YOL}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
