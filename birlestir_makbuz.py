# -*- coding: utf-8 -*-
"""PAYLI OLCUM MAKBUZLARINI BIRLESTIR.

Kullanim:  python birlestir_makbuz.py <cikti.json> <pay1.json> <pay2.json> ...

Mikro F1 parca basina TP/FP/FN toplami oldugu icin birlestirme KAYIPSIZDIR;
paylara bolup sonra toplamak tek islemde kosmakla AYNI sayiyi verir. Bu betik
ayrica ayni parcanin iki payda birden puanlanip puanlanmadigini KONTROL EDER --
oyle bir cakisma sessizce sayilari sisirirdi.
"""
import collections
import json
import sys

import numpy as np


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cik, paylar = sys.argv[1], sys.argv[2:]
    kir = {}
    cakisan = []
    damga = None
    bayrak = {}
    for y in paylar:
        m = json.load(open(y))
        damga = damga or m.get("damga")
        s = m["sonuc"]
        for k in ("p6_acik", "genis_acik", "poz_kafasi"):
            if k in s:
                bayrak.setdefault(k, s[k])
                if bayrak[k] != s[k]:
                    sys.exit(f"PAYLAR FARKLI AYARDA: {k} {bayrak[k]} vs {s[k]}")
        for p, v in (s.get("parca_kirilim") or s.get("parca_tp_fp_fn") or
                     {}).items():
            if p in kir:
                cakisan.append(p)
            kir[p] = v
    if cakisan:
        print(f"UYARI: {len(cakisan)} parca birden fazla payda puanlanmis "
              f"(ilk 5: {cakisan[:5]}). Paylar ORTUSUYOR.")

    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = [0, 0, 0]
    for p, v in kir.items():
        a = rob[v["mfg"]]
        for i in range(3):
            a[i] += v["rob"][i]
            tes[i] += v["tes"][i]
    pm = {m: f1(*a) for m, a in rob.items()}
    T = [sum(a[i] for a in rob.values()) for i in range(3)]
    out = {"robot": f1(*T), "tespit": f1(*tes),
           "makro": float(np.mean(list(pm.values()))),
           "en_kotu": float(min(pm.values())), "marka": pm,
           "TP": T[0], "FP": T[1], "FN": T[2],
           "n_parca": len(kir), "parca_kirilim": kir, **bayrak}
    json.dump({"damga": damga, "sonuc": out,
               "not": f"{len(paylar)} paydan birlestirildi. Mikro F1 parca "
                      "basina toplam oldugu icin birlestirme kayipsizdir."},
              open(cik, "w"), indent=1)
    print(f"{len(kir)} parca | robot {out['robot']:.4f} | tespit "
          f"{out['tespit']:.4f} | makro {out['makro']:.4f} | "
          f"TP {T[0]} FP {T[1]} FN {T[2]}")
    print(f"makbuz -> {cik}")


if __name__ == "__main__":
    main()
