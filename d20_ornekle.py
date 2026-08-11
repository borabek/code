# -*- coding: utf-8 -*-
"""D20 ORNEKLEME: disk sinirina sigan, AGIZ KALITESI DUSUK oncelikli alt kume.

OLCULDU (2026-08-09): operator onbellegi parca basina ~2.8 MB (k_eig=96).
5391 parca -> ~15 GB; makinede sigmadi. Bu betik D20'den ~N parca secer.

SECIM RASTGELE DEGIL: yanal hata AGIZ KALITESIYLE aciklaniyor
([[yanal-hata-segmentasyon-kalitesiyle-aciklanir]]: dusuk ceyrekte <=2mm %25.8,
yuksek ceyrekte %75.9). Kazanc DUSUK KALITELI agizlarda; ogrenilecek sey orada.

VEKIL OLCUT (etiketten hesaplanir, modele ihtiyac YOK):
  boyanan tepe sayisi / CP sayisi  -- dusuk = agiz zayif yakalanmis
  + parca basina CP sayisi (yuksek-CP oncelikli, plan boyle diyor)
MARKA DENGESI korunur: her markadan payina dusen kadar.
"""
import collections, io, json, os, shutil, sys
import numpy as np
sys.path.insert(0, ".")
import connector3d
CE = int(connector3d.CABLE_ENTRY)

KAYNAK = "_label_ds1_obj"
HEDEF = "_label_ds1_obj_SECILI"
N = int(os.environ.get("D20_N", "1500"))


def main():
    adlar = [d for d in os.listdir(KAYNAK) if os.path.isdir(os.path.join(KAYNAK, d))]
    print(f"kaynak {len(adlar)} parca | hedef ~{N}")
    kayit = []
    for ad in adlar:
        lf = os.path.join(KAYNAK, ad, ad + ".labels.txt")
        if not os.path.exists(lf):
            continue
        L = np.fromstring(io.open(lf, encoding="utf-8").read(), sep=" ", dtype=int)
        if not len(L):
            continue
        n_ce = int((L == CE).sum())
        if n_ce == 0:
            continue
        mfg = ad.split(".", 1)[0]
        kayit.append({"ad": ad, "mfg": mfg, "n_ce": n_ce, "n_v": len(L),
                      "oran": n_ce / max(len(L), 1)})
    print(f"okunabilen {len(kayit)}")
    # DUSUK 'oran' = agiz zayif yakalanmis -> ONCELIKLI
    say = collections.Counter(k["mfg"] for k in kayit)
    pay = {m: max(1, int(N * c / len(kayit))) for m, c in say.items()}
    grup = collections.defaultdict(list)
    for k in kayit:
        grup[k["mfg"]].append(k)
    sec = []
    for m, lst in grup.items():
        lst.sort(key=lambda x: x["oran"])          # dusuk oran ONCE
        sec += lst[:pay[m]]
    os.makedirs(HEDEF, exist_ok=True)
    for k in sec:
        h = os.path.join(HEDEF, k["ad"])
        if not os.path.exists(h):
            shutil.copytree(os.path.join(KAYNAK, k["ad"]), h)
    d = collections.Counter(k["mfg"] for k in sec)
    print(f"SECILDI {len(sec)} parca -> {HEDEF}")
    print(f"marka dagilimi: {dict(d)}")
    print(f"secilen medyan agiz orani {np.median([k['oran'] for k in sec]):.4f} | "
          f"TUM korpus medyani {np.median([k['oran'] for k in kayit]):.4f}")
    json.dump({"n": len(sec), "marka": dict(d), "N_hedef": N,
               "olcut": "agiz orani (boyanan tepe / toplam tepe) DUSUK oncelikli"},
              io.open("results/d20_ornek.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
