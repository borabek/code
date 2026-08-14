# -*- coding: utf-8 -*-
"""D5-4b: GORULMEMIS URETICI SINAVI -- ilk kez CIDDI OLCEKTE olculuyor.

[[gate-uretici-disi-cokusu]]: gate gorulmemis ureticide 0.7422 -> 0.2799 cokuyordu, ama o
olcum TEK uretici uzerindeydi. D5-4 ile 514 parca / 16 uretici var; soru ilk kez ciddi
olcekte yanitlanabilir: **urun gordugu dort ureticinin disinda calisiyor mu?**

UC MODEL AYNI KUMEDE:
  DAGITILAN : canli urun gate'i (w2 ile egitildi -- bu ureticileri HIC gormedi)
  TABAN     : yalniz eski parcalar, protokol filtreli
  v3        : eski + yeni (sinav kumesi ve ikizleri CIKARILMIS)

Bu kume icin UCU DE ornekelem-DISI -- yani sayilar dogrudan kiyaslanabilir. (194'luk olcum
kumesinde boyle DEGILDI: dagitilan gate orada orneklem-ICI, bkz.
[[f2-12-ara-olcum-ve-sizinti-buyuklugu]].)

TEZ DEGISMEZ: hicbir sey egitilmez; bu bir SINAVDIR.
"""
import collections
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f2_12_veri_kolu import egit, olc

SINAV_DER = "results/_der_sinav_yeni.pkl"
SINAV_JSON = "results/d5_4_sinav_kumesi.json"
MAKBUZ = "results/d5_4b_sinav_olc.json"


def main():
    import protokol
    protokol.tez_dogrula()
    import wire_gate

    with open(SINAV_DER, "rb") as f:
        DER = pickle.load(f)
    sv = json.load(io.open(SINAV_JSON, encoding="utf-8"))
    c = collections.Counter(r["mfg"] for r in DER)
    print(f"SINAV KUMESI: {len(DER)} parca / {len(c)} uretici (muhur {sv['sha16']})")
    print(f"  {dict(c.most_common())}\n")

    MOD = {}
    g = json.load(io.open("cp_config.json", encoding="utf-8"))["current_product"]["wire_gate"]
    with open(g["yol"], "rb") as f:
        MOD["DAGITILAN (canli urun)"] = pickle.load(f)
    for ad, npz in (("TABAN (yalniz eski)", "results/zengin_parite_v3_taban.npz"),
                    ("v3 (eski + YENI)", "results/zengin_parite_v3.npz")):
        m, bilgi = egit(npz)
        MOD[ad] = m
        print(f"  {ad}: korpus {bilgi['parca']} parca / {bilgi['uretici']} uretici egitildi")

    S = {}
    print(f"\n{'model':<24}{'F1':>9}{'TP':>7}{'FP':>7}{'FN':>7}")
    for ad, m in MOD.items():
        r = olc(m, DER)
        S[ad] = r
        print(f"{ad:<24}{r['F1']:>9.4f}{r['TP']:>7}{r['FP']:>7}{r['FN']:>7}")

    # URETICI KIRILIMI -- yalniz n>=5 olanlar karar verir
    say = collections.Counter(r["mfg"] for r in DER)
    ort = [m for m in say if say[m] >= 5]
    print(f"\n{'uretici':<8}{'parca':>7}" + "".join(f"{a[:12]:>14}" for a in MOD))
    for u in sorted(ort, key=lambda x: -say[x]):
        sat = "".join(f"{S[a]['uretici'].get(u, float('nan')):>14.4f}" for a in MOD)
        print(f"{u:<8}{say[u]:>7}{sat}")

    d = S["DAGITILAN (canli urun)"]
    print(f"\nCANLI URUN, GORULMEMIS 16 URETICIDE: F1 {d['F1']:.4f}")
    print(f"  194'luk olcum kumesinde (PXC+WEI, orneklem-ICI): 0.8536")
    print(f"  resmi manset (bekcili alt kume): 0.7584")
    print(f"  -> aradaki fark URETICI GENELLEMESININ bedeli")
    ur = [S["DAGITILAN (canli urun)"]["uretici"][u] for u in ort]
    print(f"  uretici yayilimi: en iyi {max(ur):.4f} / en kotu {min(ur):.4f} "
          f"(fark {max(ur)-min(ur):.4f})")
    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump({"sinav_sha16": sv["sha16"], "n_parca": len(DER),
                   "sonuc": {a: {k: v for k, v in r.items()} for a, r in S.items()},
                   "uretici_say": dict(say)}, f, indent=1, ensure_ascii=False)
    print(f"makbuz -> {MAKBUZ}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
