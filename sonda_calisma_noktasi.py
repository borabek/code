# -*- coding: utf-8 -*-
"""CALISMA NOKTASI TARAMASI — cevrimdisi, TAHMIN DOKUMUNDEN

NEDEN CEVRIMDISI. Her esik/adet denemesi bugune kadar 27 dakikalik TAM
ZINCIR kosusu gerektiriyordu. `sonda_zincir_esli_kiyas.py` artik tahminleri
diske dokuyor (`results/_tahmin_dokumu.json`), dolayisiyla ayni denemeler
SANIYELER icinde yapilabilir. Kalan surede kac kol denenebilecegini bu
belirliyor.

TARANAN (uretilen CP'ler SABIT, yalnizca hangilerinin TUTULDUGU degisir):
  guven esigi   : guven >= t
  oy esigi      : votes >= v
  ilk-k         : guvene gore ilk k (k sabit ya da parca capina bagli)
  ikili         : guven esigi + ilk-k tavani

RAPORLANAN: tespit / robot ISARETSIZ / robot ISARETLI, ayrica kesinlik ve
recall. Uc metrik birden verilir cunku yapilandirmadaki manset ISARETSIZ,
robot icin gecerli olcut ISARETLIDIR.

D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
from sina_kume import esle_macar   # noqa: E402

DOKUM = os.environ.get("CN_DOKUM", "results/_tahmin_dokumu.json")
YOL = os.environ.get("CN_YOL", "saha")


def olc(kayitlar, sec_fn):
    """sec_fn(guven, votes, n_gt_bilinmez) -> tutulacak indisler."""
    c = collections.Counter()
    for r in kayitlar:
        P = np.asarray(r["P"], float).reshape(-1, 3)
        D = np.asarray(r["D"], float).reshape(-1, 3)
        g = np.asarray(r["guven"], float)
        v = np.asarray(r["votes"], float)
        if len(P):
            tut = sec_fn(g, v)
            P, D = P[tut], D[tut]
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        dg = float(r["diag"])
        for ad, kw in (("isaretli", dict(isaretli=True)),
                       ("isaretsiz", dict(isaretli=False))):
            tp, fp, fn = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI,
                                    False, **kw)[:3]
            c[ad + "_tp"] += tp; c[ad + "_fp"] += fp; c[ad + "_fn"] += fn
        tp, fp, fn = esle_macar(P, D, G, Gd, dg, 0.0, 180.0, True)[:3]
        c["tespit_tp"] += tp; c["tespit_fp"] += fp; c["tespit_fn"] += fn
        c["cp"] += len(P)
    def f1(on):
        return (2 * c[on + "_tp"] /
                max(2 * c[on + "_tp"] + c[on + "_fp"] + c[on + "_fn"], 1))
    kes = c["isaretli_tp"] / max(c["isaretli_tp"] + c["isaretli_fp"], 1)
    rec = c["isaretli_tp"] / max(c["isaretli_tp"] + c["isaretli_fn"], 1)
    return {"tespit": f1("tespit"), "isaretsiz": f1("isaretsiz"),
            "isaretli": f1("isaretli"), "kesinlik": kes, "recall": rec,
            "cp": int(c["cp"])}


def main():
    d = json.load(open(DOKUM))
    kayitlar = [r for r in d if r["yol"] == YOL]
    if not kayitlar:
        sys.exit(f"{DOKUM} icinde '{YOL}' yolu yok")
    gh = np.concatenate([np.asarray(r["guven"], float)
                         for r in kayitlar if len(r["guven"])])
    print(f"{len(kayitlar)} parca | yol={YOL} | toplam CP {len(gh)}")
    print(f"guven dagilimi: min {np.nanmin(gh):.3f} ortanca "
          f"{np.nanmedian(gh):.3f} maks {np.nanmax(gh):.3f}")

    denemeler = [("TABAN (hepsi)", lambda g, v: np.arange(len(g)))]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        denemeler.append((f"guven>={t}",
                          lambda g, v, t=t: np.where(g >= t)[0]))
    for kk in (4, 6, 8, 12, 16, 24):
        denemeler.append((f"ilk-{kk}",
                          lambda g, v, kk=kk: np.argsort(-g)[:kk]))
    for t, kk in ((0.5, 12), (0.6, 12), (0.5, 8), (0.6, 8)):
        denemeler.append((
            f"guven>={t} & ilk-{kk}",
            lambda g, v, t=t, kk=kk: np.argsort(
                -np.where(g >= t, g, -np.inf))[:kk][
                    g[np.argsort(-np.where(g >= t, g, -np.inf))[:kk]] >= t]))

    print(f"\n{'deneme':<24}{'tespit':>9}{'isaretsiz':>11}{'ISARETLI':>10}"
          f"{'kesinlik':>10}{'recall':>9}{'CP':>7}")
    taban = None
    sonuc = {}
    for ad, fn in denemeler:
        r = olc(kayitlar, fn)
        if taban is None:
            taban = r
        sonuc[ad] = r
        yildiz = "  <-" if r["isaretli"] > taban["isaretli"] + 1e-9 else ""
        print(f"{ad:<24}{r['tespit']:>9.4f}{r['isaretsiz']:>11.4f}"
              f"{r['isaretli']:>10.4f}{r['kesinlik']:>10.4f}"
              f"{r['recall']:>9.4f}{r['cp']:>7}{yildiz}")
    en = max(sonuc.items(), key=lambda kv: kv[1]["isaretli"])
    print(f"\nEN IYI (robot ISARETLI): {en[0]} -> {en[1]['isaretli']:.4f} "
          f"(taban {taban['isaretli']:.4f}, fark "
          f"{en[1]['isaretli'] - taban['isaretli']:+.4f})")
    json.dump({"yol": YOL, "n_parca": len(kayitlar), "sonuc": sonuc,
               "en_iyi": {"ad": en[0], **en[1]},
               "not": "Cevrimdisi calisma noktasi taramasi; uretilen CP'ler "
                      "SABIT, yalnizca tutma kurali degisir. D7'ye BAKILMADI."},
              open(f"results/calisma_noktasi_{YOL}.json", "w"), indent=1)
    print(f"makbuz -> results/calisma_noktasi_{YOL}.json")


if __name__ == "__main__":
    main()
