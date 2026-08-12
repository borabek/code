# -*- coding: utf-8 -*-
"""ESLI TAVAN KIYASI: iki korpusu AYNI PARCALAR uzerinde karsilastir.

NEDEN. Tavan-24 korpusu (`tam4`) henuz tamamlanmadi. Yarim korpusta olculen
yonlu recall, tam korpusunkiyle KIYASLANAMAZ: biten parcalar rastgele degil,
ONCE BITEN yani daha kucuk/kolay parcalardir. Nitekim kismi tam4 olcumu 0.8952
verdi -- tavan-12'nin 0.7347'sinden cok yuksek, ama farkin ne kadari TAVANDAN
ne kadari KOLAY ALT KUMEDEN, bilinmiyor.

COZUM. Iki korpusu da YALNIZCA HER IKISINDE DE BULUNAN parcalar uzerinde olc.
Boylece alt kume etkisi ikisinde de AYNI olur ve fark yalnizca tavandan gelir.

Kabul kutusu urun metrigiyle birebir: yanal <= K.YANAL, ISARETLI aci <= K.ACI,
eksenel <= 40mm. Secici YOK -- bu bir TAVAN olcumudur.

Kullanim:  ES_A=results/_p6_oz_tam3 ES_B=results/_p6_oz_tam4 ES_ON=d6 \
           python sonda_esli_tavan.py
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K              # noqa: E402
from kos_p6_ortak import kayitlar   # noqa: E402

A_DIZ = os.environ.get("ES_A", "results/_p6_oz_tam3")
B_DIZ = os.environ.get("ES_B", "results/_p6_oz_tam4")
ON = os.environ.get("ES_ON", "d6")
EKSEN_TOL = 40.0


def pidler(diz):
    return {f[len(ON) + 1:-4] for f in os.listdir(diz)
            if f.startswith(ON + "_") and f.endswith(".npz")}


def olc(diz, pid_list, kay):
    """Doner: (gt, konum_yakalanan, yonlu_yakalanan, secenek, aday)."""
    gt = ky = yy = sec = ad = 0
    for pid in pid_list:
        r = kay.get(pid)
        if not r or not len(r.get("G", [])):
            continue
        z = np.load(f"{diz}/{ON}_{pid}.npz")
        idx = np.asarray(z["idx"], int)
        YD = np.asarray(z["YD"], float)
        P = np.asarray(z["P"], float)[idx]
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        gt += len(G)
        sec += len(idx)
        ad += len(np.unique(idx))
        if not len(P):
            continue
        d_ = P[:, None, :] - G[None, :, :]
        al = (d_ * Gd[None, :, :]).sum(-1)
        pe = np.linalg.norm(d_ - al[..., None] * Gd[None, :, :], axis=-1)
        konum = (pe <= K.YANAL) & (np.abs(al) <= EKSEN_TOL)
        an = np.degrees(np.arccos(np.clip(YD @ Gd.T, -1, 1)))     # ISARETLI
        ky += int(konum.any(0).sum())
        yy += int((konum & (an <= K.ACI)).any(0).sum())
    return gt, ky, yy, sec, ad


def main():
    pa, pb = pidler(A_DIZ), pidler(B_DIZ)
    ortak = sorted(pa & pb)
    print(f"A={A_DIZ} ({len(pa)} parca)  B={B_DIZ} ({len(pb)} parca)")
    print(f"ORTAK: {len(ortak)} parca -- kiyas YALNIZ bunlar uzerinde\n")
    if not ortak:
        sys.exit("ortak parca yok")
    kay = kayitlar(ortak)
    out = {}
    print(f"{'korpus':<26}{'GT':>7}{'konum':>9}{'YONLU':>9}"
          f"{'secenek/parca':>15}")
    for ad_, dz in (("A (tavan 12)", A_DIZ), ("B (tavan 24)", B_DIZ)):
        gt, ky, yy, sec, adn = olc(dz, ortak, kay)
        kr, yr = ky / max(gt, 1), yy / max(gt, 1)
        out[dz] = {"gt": gt, "konum_recall": kr, "yonlu_recall": yr,
                   "f1_tavani": 2 * yr / (1 + yr),
                   "secenek_parca": sec / max(len(ortak), 1),
                   "aday_parca": adn / max(len(ortak), 1)}
        print(f"{ad_:<26}{gt:>7}{kr:>9.4f}{yr:>9.4f}"
              f"{sec / max(len(ortak), 1):>15.0f}")
    a, b = out[A_DIZ], out[B_DIZ]
    print(f"\nFARK (B - A):  konum {b['konum_recall'] - a['konum_recall']:+.4f}"
          f"   YONLU {b['yonlu_recall'] - a['yonlu_recall']:+.4f}"
          f"   F1 tavani {b['f1_tavani'] - a['f1_tavani']:+.4f}")
    print(f"secenek maliyeti: "
          f"{b['secenek_parca'] / max(a['secenek_parca'], 1e-9):.2f}x")
    json.dump({"on": ON, "n_ortak": len(ortak), "A": A_DIZ, "B": B_DIZ,
               "sonuc": out,
               "not": "ESLI kiyas: yalniz IKI korpusta da bulunan parcalar. "
                      "Yarim korpusu tam korpusla kiyaslamak alt kume "
                      "yanliligi uretir; bu olcum onu kaldirir. D7'ye "
                      "BAKILMADI."},
              open(f"results/esli_tavan_{ON}.json", "w"), indent=1)
    print(f"\nmakbuz -> results/esli_tavan_{ON}.json")


if __name__ == "__main__":
    main()
