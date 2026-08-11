# -*- coding: utf-8 -*-
"""TIER COKUSU: "GLB'deki kirmizi (AUTO) isaretlerin kaci dogru?"

BULGU (2026-08-12). Dagitilan AUTO esiginde (`cp_config.robot_auto_gate_threshold`
= 0.6) GORULMEMIS MARKADA isaretlerin **%100'u AUTO** oluyor; REVIEW katmani
BOS kaliyor. Yani iki katmanli guvenlik mekanizmasi ATIL: robot her isarete
kendi basina guveniyor, oysa isaretlerin ancak ucte biri dogru.

NEDEN COKUYOR. Secim kurali ile tier esigi AYNI skoru kullaniyor. Secim kurali
zaten goreli (parca-maksimumunun %85'i) oldugu icin hayatta kalan her tahminin
skoru yuksek; tier esigi hicbir seyi elemiyor. Esik "baglamiyor".

TARIHCE. Ayni cokus 2026-07-29'da bir kez yasanmis ve duzeltilmisti (o zaman
tier SEGMENTASYON guvenine bakiyordu, REVIEW bos cikmisti, kesinlik 0.7735).
Duzeltme skoru degistirdi ama COKUS BICIMI geri geldi -- bu sefer gorulmemis
marka kosulunda ve cok daha dusuk kesinlikle.

BU BETIK YENI BIR D7 OKUMASI DEGILDIR: harcanmis olcumun makbuzlarini yeniden
okur, model secimi/ayar yapmaz.

Kullanim:  python sonda_tier_cokusu.py
"""
import glob
import json
import os
import sys

import numpy as np

ESIKLER = (0.0, 0.3, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9, 0.95)


def cift(y):
    """makbuz -> (skorlar, dogru, GT sayisi)."""
    try:
        d = json.load(open(y, encoding="utf-8"))
    except Exception:
        return None
    s = d.get("sonuc", {})
    kir = s.get("parca_kirilim") or s.get("parca_tp_fp_fn")
    if not kir:
        return None
    S, Y, gt = [], [], 0
    for v in kir.values():
        gt += v["rob"][0] + v["rob"][2]
        sk = v.get("skor") or []
        dg = v.get("dogru") or []
        if sk and len(sk) == len(dg):
            S += list(sk)
            Y += list(dg)
    if not S:
        return None
    return np.asarray(S, float), np.asarray(Y, bool), gt, len(kir)


def tablo(ad, S, Y, gt):
    print(f"\n### {ad}   ({len(S)} isaret, {gt} GT)")
    print(f"{'esik':>6}{'AUTO':>8}{'AUTO pay':>10}{'kesinlik':>10}"
          f"{'GT kapsama':>12}")
    sat = []
    for t in ESIKLER:
        m = S >= t
        if not m.sum():
            continue
        r = {"esik": t, "n": int(m.sum()), "pay": float(m.mean()),
             "kesinlik": float(Y[m].mean()),
             "gt_kapsama": float(Y[m].sum() / max(gt, 1))}
        sat.append(r)
        print(f"{t:>6.2f}{r['n']:>8d}{r['pay']:>10.4f}{r['kesinlik']:>10.4f}"
              f"{r['gt_kapsama']:>12.4f}")
    return sat


def main():
    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    dag = float(cfg.get("robot_auto_gate_threshold", 0.6))
    print(f"DAGITILAN AUTO ESIGI = {dag}")
    out = {"dagitilan_esik": dag, "kumeler": {}}
    for y in sorted(glob.glob("results/d7_p6.json") +
                    glob.glob("results/d7_taban.json")):
        r = cift(y)
        if not r:
            continue
        S, Y, gt, npar = r
        ad = os.path.basename(y).replace(".json", "")
        # VARSAYILAN DOLGU TUZAGI: `sonda_dagitim_dogrula` skoru
        # `c.get("wire_score", 1.0)` ile okuyor. Zincir wire_score URETMIYORSA
        # her tahmine 1.0 yazilir ve tablo "her esikte %100 AUTO" gibi gorunur.
        # Bu bir BULGU DEGIL, olcum bosllugudur -- ayirt edilmezse tier cokusu
        # diye raporlanir.
        if len(np.unique(S)) == 1 and float(S[0]) == 1.0:
            print(f"\n### {ad}  ({npar} parca) -- OLCULMEMIS")
            print(f"  {len(S)} tahminin HEPSI tam 1.0: zincir wire_score "
                  f"uretmemis, sonda varsayilani yazmis. Bu kumede tier "
                  f"davranisi hakkinda HICBIR SEY SOYLENEMEZ.")
            out["kumeler"][ad] = {"n_parca": npar, "n_isaret": int(len(S)),
                                  "durum": "OLCULMEMIS_varsayilan_dolgu"}
            continue
        sat = tablo(f"{ad}  ({npar} parca)", S, Y, gt)
        print(f"  skor dagilimi: min {S.min():.4f}  medyan "
              f"{np.median(S):.4f}  maks {S.max():.4f}")
        if S.min() >= dag:
            print(f"  !! DAGITILAN ESIK ({dag}) SKOR TABANININ ALTINDA "
                  f"({S.min():.4f}) -- esik hicbir seyi elemiyor.")
        m = S >= dag
        out["kumeler"][ad] = {
            "n_parca": npar, "n_isaret": int(len(S)), "gt": int(gt),
            "dagitilan_esikte": {
                "auto_pay": float(m.mean()),
                "kesinlik": float(Y[m].mean()) if m.sum() else None,
                "review_bos": bool(m.mean() >= 0.999)},
            "egri": sat}
        if m.mean() >= 0.999:
            print(f"  !! REVIEW KATMANI BOS: isaretlerin %100'u AUTO, "
                  f"kesinlik {Y[m].mean():.4f}")

    json.dump(out, open("results/tier_cokusu_d7.json", "w"), indent=1)
    print("\nmakbuz -> results/tier_cokusu_d7.json")
    print("\nNOT: yeni D7 OKUMASI DEGIL -- harcanmis olcumun yeniden analizi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
