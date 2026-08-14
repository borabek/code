# -*- coding: utf-8 -*-
"""GUVEN ARALIGI — parca duzeyi bootstrap, tahmin dokumundan

Sunumda tek bir F1 sayisi vermek yetmez: 100 parcalik bir kumede F1'in
belirsizligi kucuk degildir. Bu betik dokumden PARCA duzeyi bootstrap ile
%95 GA verir.

PARCA duzeyi (secenek duzeyi DEGIL) cunku ayni parcanin CP'leri bagimsiz
degildir. Projede ayrica GEOMETRIK IKIZ sizintisi olculdu (%80 ikiz), o
yuzden gercek GA burada hesaplanandan GENIS olabilir -- bu sinirlilik
raporlanir.
"""
import json, os, sys
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K
from sina_kume import esle_macar

DOKUM = os.environ.get("GA_DOKUM", "results/_dokum_taban.json")
YOL = os.environ.get("GA_YOL", "saha")
N_BOOT = int(os.environ.get("GA_N", "2000"))

def parca_sayilari(r):
    P = np.asarray(r["P"], float).reshape(-1, 3)
    D = np.asarray(r["D"], float).reshape(-1, 3)
    G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
    dg = float(r["diag"]); out = {}
    for ad, im in (("isaretli", True), ("isaretsiz", False)):
        out[ad] = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI, False,
                             isaretli=im)[:3]
    out["tespit"] = esle_macar(P, D, G, Gd, dg, 0.0, 180.0, True)[:3]
    return out

def main():
    d = [r for r in json.load(open(DOKUM)) if r["yol"] == YOL]
    if not d:
        sys.exit(f"{DOKUM} icinde '{YOL}' yok")
    say = [parca_sayilari(r) for r in d]
    rng = np.random.default_rng(0)
    print(f"{len(d)} parca | {N_BOOT} bootstrap | yol={YOL}\n")
    print(f"{'metrik':<12}{'F1':>9}{'%95 GA':>22}")
    out = {}
    for m_ in ("tespit", "isaretsiz", "isaretli"):
        t = np.array([s[m_] for s in say], float)   # (n,3) tp fp fn
        tam = t.sum(0)
        f1 = 2*tam[0]/max(2*tam[0]+tam[1]+tam[2], 1)
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(t), len(t))
            s = t[i].sum(0)
            bs.append(2*s[0]/max(2*s[0]+s[1]+s[2], 1))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[m_] = {"f1": float(f1), "ga": [float(lo), float(hi)]}
        print(f"{m_:<12}{f1:>9.4f}   [{lo:.4f}, {hi:.4f}]")
    print("\nNOT: parca duzeyi bootstrap. Projede %80 GEOMETRIK IKIZ")
    print("     olculdu; gercek GA bundan GENIS olabilir.")
    json.dump({"yol": YOL, "n_parca": len(d), "n_boot": N_BOOT, "sonuc": out,
               "not": "Parca duzeyi bootstrap %95 GA. Geometrik ikiz "
                      "sizintisi nedeniyle gercek GA daha genis olabilir."},
              open(f"results/guven_araligi_{YOL}.json", "w"), indent=1)
    print(f"makbuz -> results/guven_araligi_{YOL}.json")

if __name__ == "__main__":
    main()
