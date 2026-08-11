# -*- coding: utf-8 -*-
"""SONDA: SERBEST-DERINLIK YON YELPAZESI havuz tavanini ne kadar acar?

FIZIK: gercek giris yonu, agizdan DISARI dogru en uzun bos yolu olan yondur
(tel oradan gelir). Bu, markadan bagimsiz ve mesh-only bir YON ONERI kaynagidir.

OLCUM SORUSU (D6, seyreltilmis havuz):
  yonlu recall  BANKA           0.7264   (kendi+komsu+silindir+ana)
  yonlu recall  BANKA + YELPAZE   ?      (aday basina +K yon onerisi)
Konum-tek recall 0.8713 -> yon kaybi 0.145. Yelpaze bunun kacini geri alir?

YONTEM: her aday icin fibonacci-kure 64 yon; p + eps*d yonunde ilk carpisma
mesafesi (topakli isin atisi, `agiz_tanimlayici._ilk_mesafe`); en derin K yon
oneri olur. Olcut ISARETLI aci <= 10 (urun metrigiyle ayni).
"""
import json
import os
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import agiz_tanimlayici as AT      # noqa: E402
import d6_kayit                    # noqa: E402
import yon_bankasi as YB           # noqa: E402

YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
N = int(os.environ.get("SONDA_N", "100"))
K = int(os.environ.get("FAN_K", "3"))
NFAN = int(os.environ.get("FAN_N", "64"))
OB = "results/_p1_olasilik"
P6D = "results/_p6_oz_u25"


def fibonacci_kure(n):
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    th = phi * i
    return np.stack([np.cos(th) * r, y, np.sin(th) * r], axis=1)


def recall(P, D, G, Gd):
    if not len(P) or not len(G):
        return np.zeros(len(G), bool)
    Dn = YB.birim(D)
    Gn = YB.birim(Gd)
    df = np.asarray(P, float)[:, None, :] - np.asarray(G, float)[None, :, :]
    al = (df * Gn[None, :, :]).sum(-1)
    yan = np.linalg.norm(df - al[..., None] * Gn[None, :, :], axis=-1)
    an = np.degrees(np.arccos(np.clip(Dn @ Gn.T, -1.0, 1.0)))
    return ((yan <= YANAL) & (np.abs(al) <= EKSENEL) & (an <= ACI)).any(0)


def main():
    import trimesh
    kay = d6_kayit.yukle()
    # MARKA SUZGECI: kaldirac BOSLUGUN OLDUGU yerde sinanmali. Ilk 100 D6
    # parcasinda banka zaten 0.9293 veriyor ve olculecek pay yok; NIT'te banka
    # 0.5254 -- yeni bir yon kaynagi ancak orada anlam tasir.
    _m = os.environ.get("FAN_MARKA")
    fs = [f for f in sorted(os.listdir(P6D))
          if f.startswith("d6_") and f.endswith(".npz")
          and (not _m or (kay.get(f[3:-4]) or {}).get("mfg") == _m)][:N]
    FAN = fibonacci_kure(NFAN)
    n_gt = 0
    t_bank = t_fan = t_ikisi = 0
    n_sec_b = n_sec_f = 0
    npar = 0
    for i, f in enumerate(fs, 1):
        pid = f[3:-4]
        r = kay.get(pid)
        mf = f"{OB}/{pid}.npz"
        if r is None or not len(r.get("G", [])) or not os.path.exists(mf):
            continue
        z = np.load(f"{P6D}/{f}")
        P = np.asarray(z["P"], float)
        idx = np.asarray(z["idx"], int)
        YD = np.asarray(z["YD"], float)
        zz = np.load(mf)
        V = np.ascontiguousarray(zz["V"], np.float64)
        Fc = np.ascontiguousarray(zz["F"], np.int64)
        mesh = trimesh.Trimesh(V, Fc, process=False)
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        eps = max(1e-3, 1e-4 * diag)
        # yelpaze: her aday x 64 yon icin DISARI serbest yol
        nc = len(P)
        O = np.repeat(P, NFAN, axis=0) + eps * np.tile(FAN, (nc, 1))
        Dv = np.tile(FAN, (nc, 1))
        der = AT._ilk_mesafe(mesh, O, Dv, diag).reshape(nc, NFAN)
        top = np.argsort(-der, axis=1)[:, :K]
        Pf = np.repeat(P, K, axis=0)
        Df = FAN[top.reshape(-1)]
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        n_gt += len(G)
        hb = recall(P[idx], YD, G, Gd)
        hf = recall(Pf, Df, G, Gd)
        t_bank += int(hb.sum())
        t_fan += int(hf.sum())
        t_ikisi += int((hb | hf).sum())
        n_sec_b += len(idx)
        n_sec_f += len(Pf)
        npar += 1
        if i % 25 == 0:
            print(f"  {i}/{len(fs)}", flush=True)

    def f1t(t):
        rc = t / max(n_gt, 1)
        return rc, 2 * rc / (1 + rc)

    out = {}
    print(f"\nD6 {npar} parca | {n_gt} GT | yelpaze {NFAN} yon, en derin {K}")
    for ad, t, ns in (("BANKA", t_bank, n_sec_b), ("YELPAZE", t_fan, n_sec_f),
                      ("BANKA+YELPAZE", t_ikisi, n_sec_b + n_sec_f)):
        rc, f1 = f1t(t)
        out[ad] = {"recall": rc, "f1_tavani": f1,
                   "secenek_parca": ns / max(npar, 1)}
        print(f"{ad:<14} yonlu recall {rc:.4f}  F1 tavani {f1:.4f}  "
              f"secenek/parca {ns / max(npar, 1):.0f}")
    print(f"\nYELPAZE KAZANCI: {out['BANKA+YELPAZE']['recall'] - out['BANKA']['recall']:+.4f} recall")
    json.dump({"damga": makbuz_hash.damga(), "n_parca": npar, "n_gt": n_gt,
               "fan_n": NFAN, "fan_k": K, "sonuc": out,
               "not": "Serbest-derinlik yon yelpazesi TAVAN sondasi. D6, "
                      "seyreltilmis havuz (_p6_oz_u25), ISARETLI aci."},
              open("results/yon_fani_d6.json", "w"), indent=1)
    print("makbuz -> results/yon_fani_d6.json")


if __name__ == "__main__":
    main()
