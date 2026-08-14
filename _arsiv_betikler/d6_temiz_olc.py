# -*- coding: utf-8 -*-
"""D6-6b: TEMIZ gorulmemis-uretici sinavinda DURUST TABAN.

NEDEN: eski sinav kumesi (`d5_4_sinav_kumesi.json`) URETICI duzeyinde kirlenmisti --
seg oto-etiket korpusu o kumenin 250 parcasindan 169'unu ve sinav ureticilerinden 958
parcayi iceriyordu. Orada olculen 0.6050 SISIK. Yeni kume (`d6_sinav_kumesi.json`,
468 parca / 8 uretici) hicbir egitim yapitinda GECMEYEN ureticilerden kuruldu.

YENIDEN TURETME YOK: turetme kayitlari aday noktalarini (P), yonlerini (Pd) ve gate
oznitelik matrislerini (X, XR) zaten tasiyor. Gate karari ve iki metrik BUNLARDAN
hesaplanir -- urunun KENDI karar kodu (`wire_gate.karar_skoru` + goreli esik) cagrilir,
elde yeniden kurulmaz ([[olcum-zaafiyetleri-kapatildi]]).

IKI METRIK AYRI ([[brep-axis-and-two-metrics]]):
  TESPIT    : yanal <= max(3mm, %6*kosegen), aci SERBEST, eksenel <= 40mm
  ROBOT     : yanal <= 2mm, aci <= 10 derece, ISARETLI yon
"""
import collections
import glob
import io
import json
import os
import pickle
import sys

import numpy as np

import d6_kayit

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KUME = "results/d6_sinav_kumesi.json"
MAKBUZ = "results/d6_temiz_olc.json"
ROBOT_YANAL = 2.0
ROBOT_ACI = 10.0


def main():
    import protokol
    protokol.tez_dogrula()
    import wire_gate
    from sina_kume import esle_detay, f1w

    sv = json.load(io.open(KUME, encoding="utf-8"))
    PID = set(sv["pidler"])
    print(f"TEMIZ SINAV: {sv['n_parca']} parca | {len(sv['uretici'])} uretici | "
          f"GT {sv['gt_toplam']} CP | muhur {sv['sha16']}")
    print(f"  uretici: {sv['uretici']}\n")

    import d6_kayit
    kayit = d6_kayit.yukle(PID)
    print(f"turetme kaydi bulunan: {len(kayit)}/{len(PID)}")

    with open("results/wire_gate.pkl", "rb") as f:
        gate = pickle.load(f)
    print(f"gate: n_feat={gate['n_feat']} donusum={gate.get('donusum')}")

    T, R = [], []                       # (rejim, tp, fp, fn)
    Tm = collections.defaultdict(list); Rm = collections.defaultdict(list)
    atlanan = collections.Counter()
    for pid, r in kayit.items():
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        P = np.zeros((0, 3)); D = np.zeros((0, 3))
        if r.get("X") is not None:
            # DAGITILAN gate 58 sutunla egitildi ve parca-ici z-skor onu 116'ya cikarir.
            # (X + XR birlestirmek 94 sutun verir -> 188 != 116 ve HER parca sessizce
            #  elenir; ilk kosuda tam da bu oldu, F1 0.0018 cikti.)
            M = d6_kayit.x58(r)
            if M is not None and M.shape[1] * 2 == gate["n_feat"]:
                k = wire_gate.karar_maskesi(wire_gate.karar_skoru(gate, M))
                if k.any():
                    P = np.asarray(r["P"], float)[k]; D = np.asarray(r["Pd"], float)[k]
            else:
                atlanan["genislik"] += 1
        else:
            atlanan["aday_yok"] += 1
        # TESPIT: aci serbest, isaretsiz
        tp, fp, fn, _ = esle_detay(P, D, G, Gd, r["diag"], 0.0, 180.0, True)
        T.append((rj, tp, fp, fn)); Tm[r["mfg"]].append((rj, tp, fp, fn))
        # ROBOT: yanal 2mm SABIT (pct=False -- pct=True 'tol'u YOK SAYAR ve ilk kosuda
        # tam bu hatayi yaptim: 0.2353 aslinda "tespit toleransi + aci<=10" idi),
        # aci 10 derece, ISARETLI. Resmi cagri s0s4_triyaj.py:103 ile BIREBIR.
        tp2, fp2, fn2, _ = esle_detay(P, D, G, Gd, r["diag"], ROBOT_YANAL, ROBOT_ACI,
                                      False, isaretli=True)
        R.append((rj, tp2, fp2, fn2)); Rm[r["mfg"]].append((rj, tp2, fp2, fn2))

    if atlanan:
        print(f"ATLANAN: {dict(atlanan)}")
    tf = f1w(T); rf = f1w(R)
    print(f"\n{'':<8}{'TESPIT':>10}{'ROBOT':>10}")
    print(f"{'TOPLAM':<8}{tf:>10.4f}{rf:>10.4f}\n")
    print(f"{'uretici':<8}{'n':>5}{'TESPIT':>10}{'ROBOT':>10}")
    for m in sorted(Tm, key=lambda x: -len(Tm[x])):
        print(f"{m:<8}{len(Tm[m]):>5}{f1w(Tm[m]):>10.4f}{f1w(Rm[m]):>10.4f}")
    yay = [f1w(Tm[m]) for m in Tm if len(Tm[m]) >= 5]
    print(f"\nURETICI YAYILIMI (n>=5): en kotu {min(yay):.4f} | en iyi {max(yay):.4f}")

    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump({"kume": KUME, "muhur": sv["sha16"], "n": len(kayit),
                   "tespit": tf, "robot": rf,
                   "uretici_tespit": {m: f1w(v) for m, v in Tm.items()},
                   "uretici_robot": {m: f1w(v) for m, v in Rm.items()},
                   "not": ("TEMIZ kume. Eski d5_4 kumesindeki sayilarla KIYASLANAMAZ -- "
                           "o kume uretici duzeyinde kirliydi.")}, f, indent=1,
                  ensure_ascii=False)
    print(f"makbuz -> {MAKBUZ}")


if __name__ == "__main__":
    main()
