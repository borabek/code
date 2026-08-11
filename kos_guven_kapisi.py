# -*- coding: utf-8 -*-
"""GUVEN KAPISI: robotun kullandigi isaretlerin kesinligi >= 0.90 olsun.

SAHA GERCEGI: akis iki rejimin karisimi -- bilinen markadan yeni model + hic
bilinmeyen markadan yeni model. Tam otomatik tek sayi 0.90 bu karisimda
VERILEMEZ (tavan ~0.84-0.86). Verilebilecek soz sudur:

    "Robotun kullandigi isaretler >=0.90 kesinliktedir;
     bu kalitede otomatik KAPSAMA su an %X'tir."

Iki sayi AYRI raporlanir, asla tek bir 0.90'a karistirilmaz.

YONTEM: her tahminin skoru ve dogru olup olmadigi olcum makbuzunda duruyor
(`parca_kirilim[pid]["skor"] / ["dogru"]`). Skor esigi t icin:
    kesinlik(t) = dogru(skor>=t) / tahmin(skor>=t)
    kapsama(t)  = tahmin(skor>=t) / tum tahminler        (isaret kapsamasi)
    GT kapsama  = dogru(skor>=t) / GT                     (islenen CP orani)
En kucuk t secilir ki kesinlik >= HEDEF olsun.

KALIBRASYON / RAPOR AYRIMI: esik KALIBRASYON makbuzunda secilir (egitim ya da
gelistirme kumesi), SINAV makbuzunda yalnizca UYGULANIR. Sinavda esik aramak,
sinavdan ayar cekmek olurdu.

Kullanim:
    python kos_guven_kapisi.py <kalibrasyon.json> [sinav.json] [hedef]
"""
import json
import os
import sys

import numpy as np

HEDEF = 0.90


def cift(makbuz):
    """Makbuzdan (skor, dogru) dizileri + GT sayisi."""
    s = makbuz["sonuc"]
    kir = s.get("parca_kirilim") or {}
    S, Y = [], []
    n_gt = 0
    for v in kir.values():
        sk = v.get("skor")
        dg = v.get("dogru")
        n_gt += v["rob"][0] + v["rob"][2]        # TP + FN
        if not sk or dg is None or len(sk) != len(dg):
            continue
        S += list(sk)
        Y += list(dg)
    return np.asarray(S, float), np.asarray(Y, int), n_gt


def egri(S, Y, n_gt):
    """Azalan esik boyunca (esik, kesinlik, isaret_kapsama, gt_kapsama)."""
    if not len(S):
        return []
    i = np.argsort(-S)
    s, y = S[i], Y[i]
    dog = np.cumsum(y)
    n = np.arange(1, len(s) + 1)
    return list(zip(s, dog / n, n / len(s), dog / max(n_gt, 1)))


def esik_sec(S, Y, n_gt, hedef):
    """Kesinligi >= hedef tutan EN DUSUK esik (yani en genis kapsama)."""
    e = egri(S, Y, n_gt)
    iyi = [t for t in e if t[1] >= hedef]
    return iyi[-1] if iyi else None


def bas(ad, S, Y, n_gt, esik=None, hedef=HEDEF):
    print(f"\n--- {ad} ---")
    if not len(S):
        print("  skor/dogru verisi YOK (makbuz eski surumle uretilmis)")
        return None
    print(f"  tahmin {len(S)} | dogru {int(Y.sum())} | GT {n_gt} | "
          f"ham kesinlik {Y.mean():.4f}")
    if esik is None:
        r = esik_sec(S, Y, n_gt, hedef)
        if r is None:
            print(f"  !! kesinlik hicbir esikte >= {hedef:.2f} olmuyor "
                  f"(en yuksek {max(t[1] for t in egri(S, Y, n_gt)):.4f})")
            return None
        esik = float(r[0])
        print(f"  SECILEN esik {esik:.4f} (kalibrasyon)")
    k = S >= esik
    if not k.any():
        print(f"  esik {esik:.4f} hicbir tahmini gecirmiyor")
        return esik
    print(f"  esik {esik:.4f} -> KESINLIK {Y[k].mean():.4f} | "
          f"isaret kapsamasi {k.mean():.4f} | "
          f"GT kapsamasi {Y[k].sum() / max(n_gt, 1):.4f} "
          f"({int(Y[k].sum())}/{n_gt})")
    return esik


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    kal = json.load(open(sys.argv[1]))
    hedef = float(sys.argv[3]) if len(sys.argv) > 3 else HEDEF
    S, Y, n = cift(kal)
    esik = bas(f"KALIBRASYON  {sys.argv[1]}", S, Y, n, None, hedef)

    out = {"hedef_kesinlik": hedef, "esik": esik, "kalibrasyon": sys.argv[1]}
    if esik is not None and len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        sin = json.load(open(sys.argv[2]))
        S2, Y2, n2 = cift(sin)
        bas(f"SINAV  {sys.argv[2]}  (esik UYGULANIR, aranmaz)", S2, Y2, n2,
            esik, hedef)
        if len(S2):
            k = S2 >= esik
            out["sinav"] = {
                "dosya": sys.argv[2], "kesinlik": float(Y2[k].mean()) if k.any() else 0.0,
                "isaret_kapsama": float(k.mean()),
                "gt_kapsama": float(Y2[k].sum() / max(n2, 1)),
                "onayli": int(k.sum()), "oneri": int((~k).sum())}
            print(f"\nMUHUR: robotun kullandigi isaretler "
                  f"{out['sinav']['kesinlik']:.1%} kesinlikte; bu kalitede "
                  f"otomatik kapsama GT'nin {out['sinav']['gt_kapsama']:.1%}'i "
                  f"({out['sinav']['onayli']} ONAYLI / "
                  f"{out['sinav']['oneri']} ONERI).")
    json.dump(out, open("results/guven_kapisi.json", "w"), indent=1)
    print("\nmakbuz -> results/guven_kapisi.json")


if __name__ == "__main__":
    main()
