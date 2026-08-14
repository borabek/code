# -*- coding: utf-8 -*-
"""IKI DOKUMU ESLI KIYASLA -- genel amacli.

Kullanim:
    python sonda_dokum_kiyas.py TABAN.json KOL.json [yol]

Ortak parcalarda, uc metrikte (tespit / robot isaretsiz / robot ISARETLI)
parca duzeyi ESLI bootstrap yapar. Kiyas ayni parcalarda oldugu icin
dogru test marjinal guven araligi degil FARKIN dagilimidir.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sina_kume import esle_macar                       # noqa: E402

OLCUTLER = (("tespit", 0.0, 180.0, True, False),
            ("rob", 2.0, 10.0, False, False),
            ("rbi", 2.0, 10.0, False, True))


def _birim(v):
    v = np.asarray(v, float).reshape(-1, 3)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def yukle(fp, yol):
    return {str(r["pid"]): r for r in json.load(open(fp))
            if r.get("yol") == yol}


def olc(kayitlar, pidler):
    tot = {k[0]: [0, 0, 0] for k in OLCUTLER}
    parca = []
    for pid in pidler:
        r = kayitlar[pid]
        G = np.asarray(r["G"], float).reshape(-1, 3)
        if not len(G):
            continue
        Gd = _birim(r["Gd"])
        P = np.asarray(r["P"], float).reshape(-1, 3)
        D = _birim(r["D"]) if len(P) else np.zeros((0, 3))
        satir = {}
        for ad, tol, am, pct, isr in OLCUTLER:
            tp, fp, fn, _ = esle_macar(P, D, G, Gd, float(r["diag"]),
                                       tol, am, pct, isaretli=isr)
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
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), \
        (d > 0).mean()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    fa, fb = sys.argv[1], sys.argv[2]
    yol = sys.argv[3] if len(sys.argv) > 3 else "saha"
    A, B = yukle(fa, yol), yukle(fb, yol)
    ortak = sorted(set(A) & set(B))
    print(f"yol={yol} | {os.path.basename(fa)} ({len(A)}) vs "
          f"{os.path.basename(fb)} ({len(B)}) -> ortak {len(ortak)} parca")
    if not ortak:
        print("ORTAK PARCA YOK")
        return 1
    ma, pa = olc(A, ortak)
    mb, pb = olc(B, ortak)
    print(f"\n{'':22s} {'tespit':>8s} {'rob':>8s} {'rob-ISR':>8s}")
    print(f"{'TABAN':22s} {ma['tespit']:8.4f} {ma['rob']:8.4f} "
          f"{ma['rbi']:8.4f}")
    print(f"{'KOL':22s} {mb['tespit']:8.4f} {mb['rob']:8.4f} "
          f"{mb['rbi']:8.4f}")
    print(f"\n{'metrik':>8s} {'fark':>9s} {'%95 GA':>22s} {'poz%':>6s}")
    for ad, *_r in OLCUTLER:
        f, lo, hi, pz = boot(pa, pb, ad)
        yz = " *" if (lo > 0 or hi < 0) else ""
        print(f"{ad:>8s} {f:+9.4f} [{lo:+.4f},{hi:+.4f}]{yz:>3s} "
              f"{100*pz:5.1f}")
    print("\n(* = %95 GA sifiri icermiyor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
