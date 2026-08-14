# -*- coding: utf-8 -*-
"""ESLI FARK BOOTSTRAP — bir son-islem kuralinin kazanci GERCEK mi?

`halka_disari` kurali robot ISARETLI F1'i 0.4839 -> 0.5168 (+0.0329)
cikardi. Marjinal guven araligi (±0.08) burada YANLIS testtir: kiyas AYNI
parcalarda yapiliyor, dolayisiyla anlamli olan FARKIN dagilimidir.

Bu betik parca duzeyi ESLI bootstrap yapar: her tekrarda ayni parca
kumesi secilir, iki kol da o kumede hesaplanir, FARK kaydedilir.
GA sifiri icermiyorsa kazanc gercektir.
"""
import json, os, sys
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K
from sina_kume import esle_macar
from sonda_son_islem import uygula

DOKUM = os.environ.get("EF_DOKUM", "results/_dokum_halka.json")
YOL = os.environ.get("EF_YOL", "saha")
KOL = os.environ.get("EF_KOL", "halka_disari")
N_BOOT = int(os.environ.get("EF_N", "2000"))

def say(r, kol):
    P = np.asarray(r["P"], float).reshape(-1, 3)
    D = np.asarray(r["D"], float).reshape(-1, 3)
    G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
    P, D = uygula(P.copy(), D.copy(), kol, G, Gd,
                  r.get("mesh_merkez"), r.get("yerel_normal"),
                  r.get("halka_normal"))
    return esle_macar(P, D, G, Gd, float(r["diag"]), K.YANAL, K.ACI,
                      False, isaretli=True)[:3]

def main():
    d = [r for r in json.load(open(DOKUM)) if r["yol"] == YOL]
    A = np.array([say(r, "taban") for r in d], float)
    B = np.array([say(r, KOL) for r in d], float)
    def f1(t):
        s = t.sum(0)
        return 2*s[0]/max(2*s[0]+s[1]+s[2], 1)
    fa, fb = f1(A), f1(B)
    rng = np.random.default_rng(0); bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(A), len(A))
        bs.append(f1(B[i]) - f1(A[i]))
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{len(d)} parca | kol={KOL} | {N_BOOT} ESLI bootstrap\n")
    print(f"  taban      {fa:.4f}")
    print(f"  {KOL:<10} {fb:.4f}")
    print(f"  FARK       {fb-fa:+.4f}   %95 GA [{lo:+.4f}, {hi:+.4f}]")
    kesin = lo > 0
    print(f"\n  GA sifiri ICERMIYOR mu: {'EVET -> kazanc GERCEK' if kesin else 'HAYIR -> kanit yetersiz'}")
    print(f"  bootstrap orneklerinin %{(bs>0).mean()*100:.1f}'i pozitif")
    json.dump({"kol": KOL, "n_parca": len(d), "taban": fa, "kol_f1": fb,
               "fark": fb-fa, "ga": [lo, hi], "pozitif_oran": float((bs>0).mean()),
               "kesin": bool(kesin),
               "not": "ESLI parca duzeyi bootstrap. Marjinal GA degil FARKIN "
                      "GA'si. D7'ye BAKILMADI."},
              open(f"results/esli_fark_{KOL}.json", "w"), indent=1)
    print(f"makbuz -> results/esli_fark_{KOL}.json")

if __name__ == "__main__":
    main()
