# -*- coding: utf-8 -*-
"""BILINEN MARKA / GORULMEMIS MODEL senaryosunun TAVANI.

Soru: "markasi egitimde olan ama modeli olmayan bir .stp'de robot 0.80 olur mu?"
Kayittaki 0.5978, yeniden uretemedigim 0.2344 ile AYNI kosudan geliyor -- ona
dayanarak cevap verilemez. Bu betik senaryonun TAVANINI olcer; tavan 0.80'in
altindaysa cevap zaten hayirdir ve secici tartismasina hic girilmez.

KUME: `split3.json` -> VAL (100 parca, 87 geometri grubu). LOCKED TEK ATISLIK,
bir fizibilite sorusu icin HARCANMAZ.

Olculenler (hicbiri gate kullanmaz -> gate kalitesinden BAGIMSIZ):
  havuz recall     GT'nin yuzde kaci havuzda ULASILABILIR (tespit toleransi)
  donusum          ulasilabilir GT'lerin yuzde kaci ROBOT olcutunu de saglar
  tespit tavani    mukemmel secici ile tespit F1
  robot tavani     mukemmel secici ile robot F1
Iki havuz icin ayri: TEZ-SAF (`v_o`) ve GENISLETILMIS (+B-rep).
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import brep_havuz               # noqa: E402
import kanonik_d7 as K          # noqa: E402
from sina_kume import esle_macar  # noqa: E402


def main():
    s = json.load(open("results/split3.json"))
    val = [str(p) for p in s["val"]["parts"]]
    d7 = {str(p) for p in json.load(open("results/d7_sinav_kumesi.json"))["pidler"]}
    kesisim = set(val) & d7
    print(f"VAL {len(val)} parca | D7 ile kesisim {len(kesisim)} "
          f"(senaryolar AYRI olmali)", flush=True)

    # VAL parcalari kanonik G7 BIRLESIMINDE YOK; iki turetmede de tam kapsaniyor.
    # Tavan cevabinin TURETMEYE BAGLI OLMADIGINI gostermek icin ikisi de kosulur.
    KAYITLAR = os.environ.get("VAL_KAYIT", "results/_der_yeni.pkl")
    R = {str(r["pid"]): r for r in pickle.load(open(KAYITLAR, "rb"))}
    print(f"kayit dosyasi: {KAYITLAR}", flush=True)
    var = [p for p in val if p in R and len(R[p].get("G", []))]
    print(f"kayitta bulunan ve GT'si olan: {len(var)}", flush=True)
    if not var:
        raise SystemExit("VAL parcalari kanonik kayitta YOK -- olcum yapilamaz")

    cy = ac = None
    for f in ("results/_d6_silindirler.pkl", "results/_brepegit_silindirler.pkl",
              "results/_d7_silindirler.pkl"):
        o = pickle.load(open(f, "rb"))
        cy = o if cy is None else {**o, **cy}
    for f in ("results/_d6_acikliklar.pkl", "results/_brepegit_acikliklar.pkl",
              "results/_d7_acikliklar.pkl"):
        o = pickle.load(open(f, "rb"))
        ac = o if ac is None else {**o, **ac}
    kapsam = sum(1 for p in var if cy.get(p))
    print(f"B-rep onbellegi olan VAL parcasi: {kapsam}/{len(var)}", flush=True)

    out = {}
    for ad, genis in (("TEZ-SAF (v_o)", False), ("GENISLETILMIS (+B-rep)", True)):
        TP_t = FN_t = TP_r = FN_r = 0
        per = collections.defaultdict(lambda: [0, 0])
        for pid in var:
            r = R[pid]
            G = np.asarray(r["G"], float)
            Gd = np.asarray(r["Gd"], float)
            dg = float(r["diag"])
            P = np.asarray(r["P"], float)
            D = np.asarray(r["Pd"], float)
            if genis:
                P, D, _ = brep_havuz.birlesik_havuz(P, D, cy.get(pid), ac.get(pid))
            tol = max(3.0, 0.06 * dg)
            tp, _fp, fn = esle_macar(P, D, G, Gd, dg, tol, 180.0, True)[:3]
            TP_t += tp; FN_t += fn
            tp2, _f2, fn2 = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI, False,
                                       isaretli=True)[:3]
            TP_r += tp2; FN_r += fn2
            a = per[r["mfg"]]
            a[0] += tp2; a[1] += fn2
        rec = TP_t / max(TP_t + FN_t, 1)
        rrec = TP_r / max(TP_r + FN_r, 1)
        # Mukemmel secici: secilmeyen aday CIKTI DEGIL -> FP yok, F1 = 2TP/(2TP+FN)
        t_tavan = 2 * TP_t / max(2 * TP_t + FN_t, 1)
        r_tavan = 2 * TP_r / max(2 * TP_r + FN_r, 1)
        out[ad] = {"havuz_recall": rec, "robot_recall": rrec,
                   "donusum": rrec / max(rec, 1e-9),
                   "tespit_tavani": t_tavan, "robot_tavani": r_tavan,
                   "n_parca": len(var)}
        c = out[ad]
        print(f"\n{ad}")
        print(f"  havuz recall   {rec:.4f}")
        print(f"  donusum        {c['donusum']:.4f}  (ulasilabilirin robot olani)")
        print(f"  TESPIT TAVANI  {t_tavan:.4f}")
        print(f"  ROBOT TAVANI   {r_tavan:.4f}", flush=True)

    print("\n0.80 HEDEFI:")
    for ad, c in out.items():
        for olcut, v in (("tespit", c["tespit_tavani"]), ("robot", c["robot_tavani"])):
            print(f"  {ad:<24} {olcut:<7} tavan {v:.4f} -> "
                  + ("ULASILABILIR (tavan icinde)" if v >= 0.80
                     else "IMKANSIZ (tavan altinda)"))
    json.dump({"damga": makbuz_hash.damga(), "sonuc": out, "n_val": len(var),
               "brep_kapsami": kapsam,
               "kayit": os.environ.get("VAL_KAYIT", "results/_der_yeni.pkl"),
               "not": "VAL kumesi (split3.json), MUKEMMEL secici tavani. Gate "
                      "KULLANILMADI -> gate kalitesinden bagimsiz. LOCKED "
                      "HARCANMADI."},
              open(os.environ.get("VAL_CIKTI", "results/bilinen_marka_tavani.json"), "w"), indent=1)
    print("\nmakbuz -> results/bilinen_marka_tavani.json")


if __name__ == "__main__":
    main()
