# -*- coding: utf-8 -*-
"""REJIM ESIGININ KARARLILIGI: 90 bir bicak sirti mi?

SORUN. Rejim kapisi `n01 >= 90 -> P6`. D7'de parcalarin n01 ortalamasi 89.6 --
yani esik tam dagilimin ortasinda ve kucuk bir kayma yonlendirmeyi tersine
cevirebiliyor. Nitekim CWT (n01=86) tabana, WIE (n01=141) P6'ya gidiyor ve
IKISI DE yanlis tarafta.

Esigi D7'ye bakarak oynatmak SINAVDAN AYAR CEKMEKTIR. Bunun yerine burada
`tam` korpusunun MARKA KATLARINDA esik EGRISI cikarilir:

  * her esik degeri icin kat-disi robot F1
  * egrinin DUZ oldugu bir bant var mi (kararlilik)
  * en iyi esik ile 90'in farki anlamli mi

Egri duzse esik onemsizdir ve bicak sirti korkusu yersizdir. Egri sivriyse
esik kirilgandir ve YUMUSAK GECIS (iki kolun birlesimi ya da bant icinde
tabana yaslanma) gerekir.
"""
import collections
import json
import os
import pickle
import sys
import time

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_u25")
sys.path.insert(0, ".")
import kanonik_d7 as K                       # noqa: E402
import urun_genis                            # noqa: E402
from kos_p6_ortak import yukle               # noqa: E402
from kos_p6_rejim import p6_cikti, taban_cikti  # noqa: E402
from sina_kume import esle_macar             # noqa: E402

PAKET = os.environ.get("P6_MODEL", "results/p6_kademe2_model.pkl")
ESIKLER = list(range(40, 200, 10))


def say(P, D, d):
    return esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI,
                      False, isaretli=True)[:3]


def puanla(veri, secim):
    T = [0, 0, 0]
    for d, p6 in zip(veri, secim):
        c = d["_p6_c"] if p6 else d["_tb_c"]
        for i in range(3):
            T[i] += c[i]
    return 2 * T[0] / max(2 * T[0] + T[1] + T[2], 1)


def main():
    t0 = time.time()
    pk = pickle.load(open(PAKET, "rb"))
    tb = urun_genis.model_yukle()
    veri = []
    for kume in os.environ.get("P6_KUME", "tam,d6").split(","):
        veri += yukle(kume.strip(), int(os.environ.get("P6_TR", "0")))
    print(f"{len(veri)} parca ({time.time() - t0:.0f} s)", flush=True)
    for i, d in enumerate(veri, 1):
        (Pt, Dt), _s = taban_cikti(d, tb)
        Pp, Dp = p6_cikti(d, pk)
        d["_tb_c"] = say(Pt, Dt, d)
        d["_p6_c"] = say(Pp, Dp, d)
        d["_n01"] = float((d["kaynak"] != 2).sum())
        if i % 600 == 0:
            print(f"  {i}/{len(veri)} ({time.time() - t0:.0f} s)", flush=True)

    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= 200]
    print(f"katlar: {katlar}\n", flush=True)
    print(f"{'esik':>6}{'kat-disi robot':>16}{'P6 orani':>10}")
    egri = {}
    for e in ESIKLER:
        T = [0, 0, 0]
        p6n = tot = 0
        for b in katlar:
            dis = [d for d in veri if d["mfg"] == b]
            for d in dis:
                c = d["_p6_c"] if d["_n01"] >= e else d["_tb_c"]
                for i in range(3):
                    T[i] += c[i]
                p6n += int(d["_n01"] >= e)
                tot += 1
        f1 = 2 * T[0] / max(2 * T[0] + T[1] + T[2], 1)
        egri[e] = {"robot": f1, "p6_orani": p6n / max(tot, 1)}
        print(f"{e:>6}{f1:>16.4f}{p6n / max(tot, 1):>10.2f}")

    en = max(egri, key=lambda k: egri[k]["robot"])
    tepe = egri[en]["robot"]
    bant = [e for e in ESIKLER if egri[e]["robot"] >= tepe - 0.01]
    print(f"\nEN IYI esik {en} -> {tepe:.4f}")
    print(f"TEPEDEN 0.01 ICINDE kalan esikler: {bant}")
    print(f"90'in degeri: {egri.get(90, {}).get('robot', 0):.4f} "
          f"(tepeden {tepe - egri.get(90, {}).get('robot', 0):+.4f})")
    print("YORUM: bant genisse esik KARARLI, dar ise KIRILGAN.")
    json.dump({"damga": makbuz_hash.damga(), "egri": egri, "en_iyi": en,
               "bant": bant, "katlar": katlar, "n_parca": len(veri),
               "not": "Rejim esigi kararlilik egrisi. Kat-disi olcum; D7'ye "
                      "BAKILMADI. Esik secimi bu egriden yapilir."},
              open("results/rejim_kararlilik.json", "w"), indent=1)
    print(f"makbuz -> results/rejim_kararlilik.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
