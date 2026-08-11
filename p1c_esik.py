# -*- coding: utf-8 -*-
"""P1-GATE-C: goreli esik (oran/taban) yeni aday yogunluguna gore yeniden secilir.

Dagitilan kural: aday, kendi PARCASINDAKI en yuksek skorun **0.5 katini** gecmeli VE
**0.25 tabanini** asmali. Bu iki sayi ESKI aday havuzuyla secilmisti; o zamandan beri
seg agi degisti (aday recall %31.7 -> %92.1) ve gate v5 yeniden egitildi. Otopside
GATE_REDDI hala GT'nin %10.5'i.

SECIM YANLILIGI ENGELI: 8 sinav ureticisi IKIYE bolunur.
  DEV  : SUPU, NIT, S+S, SE      (esik BURADA secilir)
  SINAV: UPUN, MOR, UTL, ONV     (secilen esik BURADA TEK ATIS olculur)
Ikisi de "gorulmemis uretici" ozelligini korur; secilen sayi ile raporlanan sayi
AYNI parcalardan gelmez. ([[uclu-bolme-ve-sahte-kazanclar]])
"""
import argparse
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

import d6_kayit

MAKBUZ = "results/p1c_esik.json"
DEV_MFG = {"SUPU", "NIT", "S+S", "SE"}
ROBOT_YANAL, ROBOT_ACI = 2.0, 10.0


def maske(skor, oran, taban):
    """Urunun goreli karar kurali -- tek kaynak burada TEKRAR EDILMEZ, ayni formul."""
    if not len(skor):
        return np.zeros(0, bool)
    return (skor >= oran * float(np.max(skor))) & (skor >= taban)


def puanla(kayit, model, oran, taban, esle_detay, f1w, mfgler=None):
    import wire_gate
    T, R = [], []
    for pid, r in kayit.items():
        if mfgler is not None and r["mfg"] not in mfgler:
            continue
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        P = np.zeros((0, 3)); D = np.zeros((0, 3))
        M = d6_kayit.x58(r)
        if M is not None and r.get("P") is not None and len(r["P"]) \
                and M.shape[1] * 2 == model["n_feat"]:
            s = wire_gate.karar_skoru(model, M)
            k = maske(np.asarray(s, float), oran, taban)
            if k.any():
                P = np.asarray(r["P"], float)[k]; D = np.asarray(r["Pd"], float)[k]
        T.append((rj,) + esle_detay(P, D, G, Gd, r["diag"], 0.0, 180.0, True)[:3])
        R.append((rj,) + esle_detay(P, D, G, Gd, r["diag"], ROBOT_YANAL, ROBOT_ACI,
                                    False, isaretli=True)[:3])
    return f1w(T), f1w(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/wire_gate_v5.pkl")
    a = ap.parse_args()
    import protokol
    protokol.tez_dogrula()
    from sina_kume import esle_detay, f1w

    sv = d6_kayit.sinav()
    kayit = d6_kayit.yukle(set(sv["pidler"]))
    with open(a.model, "rb") as f:
        model = pickle.load(f)
    dev = {p: r for p, r in kayit.items() if r["mfg"] in DEV_MFG}
    sin = {p: r for p, r in kayit.items() if r["mfg"] not in DEV_MFG}
    print(f"model: {a.model}")
    print(f"DEV   {len(dev)} parca {sorted({r['mfg'] for r in dev.values()})}")
    print(f"SINAV {len(sin)} parca {sorted({r['mfg'] for r in sin.values()})}\n")

    oranlar = [0.30, 0.40, 0.50, 0.60, 0.70]
    tabanlar = [0.15, 0.20, 0.25, 0.30, 0.35]
    print(f"{'oran/taban':<12}" + "".join(f"{t:>9.2f}" for t in tabanlar))
    en_iyi, en_iyi_skor = None, -1.0
    izgara = {}
    for o in oranlar:
        satir = []
        for t in tabanlar:
            tf, rf = puanla(dev, model, o, t, esle_detay, f1w)
            izgara[f"{o}/{t}"] = {"tespit": tf, "robot": rf}
            satir.append(tf)
            if tf > en_iyi_skor:
                en_iyi_skor, en_iyi = tf, (o, t)
        print(f"{o:<12.2f}" + "".join(f"{v:>9.4f}" for v in satir))

    o0, t0 = 0.50, 0.25
    dt0, dr0 = puanla(dev, model, o0, t0, esle_detay, f1w)
    print(f"\nDEV'de secilen: oran {en_iyi[0]:.2f} / taban {en_iyi[1]:.2f} "
          f"-> tespit {en_iyi_skor:.4f}  (mevcut 0.50/0.25: {dt0:.4f})")

    st0, sr0 = puanla(sin, model, o0, t0, esle_detay, f1w)
    st1, sr1 = puanla(sin, model, en_iyi[0], en_iyi[1], esle_detay, f1w)
    print(f"\n--- SINAV YARISI (TEK ATIS, secimde KULLANILMADI) ---")
    print(f"{'ayar':<18}{'TESPIT':>9}{'ROBOT':>9}")
    print(f"{'mevcut 0.50/0.25':<18}{st0:>9.4f}{sr0:>9.4f}")
    print(f"{f'yeni {en_iyi[0]:.2f}/{en_iyi[1]:.2f}':<18}{st1:>9.4f}{sr1:>9.4f}")
    print(f"{'FARK':<18}{st1-st0:>+9.4f}{sr1-sr0:>+9.4f}")
    karar = "DAGIT" if st1 > st0 else "GERI AL (secim DEV'e ozgu cikti)"
    print(f"\nKARAR: {karar}")
    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump({"model": a.model, "dev_mfg": sorted(DEV_MFG), "izgara": izgara,
                   "dev_secim": {"oran": en_iyi[0], "taban": en_iyi[1],
                                 "dev_tespit": en_iyi_skor},
                   "sinav_mevcut": {"tespit": st0, "robot": sr0},
                   "sinav_yeni": {"tespit": st1, "robot": sr1},
                   "karar": karar}, f, indent=1, ensure_ascii=False)
    print(f"makbuz -> {MAKBUZ}")


if __name__ == "__main__":
    main()
