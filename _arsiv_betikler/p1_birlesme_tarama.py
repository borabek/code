# -*- coding: utf-8 -*-
"""P1: CP'LERIN BIRLESMESINI ENGELLE -- en ucuz somut kaldirac.

SORUN: komsu GERCEK girisler su an uc ayri yaricapla tek adaya cokebiliyor:
  * vote pooling  `_vote2(cluster_mm=5.0)`   -- modeller arasi oy havuzu
  * component cluster `cluster_mm=3.0`       -- ayni model icinde bilesen kumeleme
  * CE-CT dedupe  `dedupe_mm=10.0`           -- sinif arasi kopya eleme
Klemens bloklarinda komsu girisler 3-5mm arayla; 10mm'lik dedupe iki gercek girisi
TEK adaya indirebilir. Bu, otopsideki KALABALIK kovasinin (GT'nin %12.3'u) ve cok-CP
rejimindeki cokusun (robot 0.0351) dogrudan supheli kaynagi.

OLCU: **ADAY KAHINI** = mukemmel gate mevcut havuzla ne yapardi (one-to-one, Macar).
Gate'i isin icine katmak iki degisikligi karistirirdi; ayrica havuz buyudugunde gate
yeniden egitilmelidir (GO gecerse ayri adim).

GO OLCUTU (kullanicinin yazdigi):
  genel kahin >= +0.01, cok-CP >= +0.015, dusuk-CP kayip <= 0.005

BIRLIKTE RAPORLANIR: kopya FP, kaybolan CP, sure, bellek.
TEZ DEGISMEZ: ayni remesh, ayni 5 sinif, ayni `v_o`; yalnizca SON ISLEM yaricaplari.
"""
import argparse
import collections
import glob
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d6_kayit

ONBELLEK = "results/_p1_olasilik"
MAKBUZ = "results/p1_birlesme_tarama.json"


def havuz(V, F, pbs, step_path, cluster_mm, dedupe_mm, oy_mm, min_v=4, vc=0.3):
    """Verilen yaricaplarla aday havuzunu URUNUN kendi koduyla uret."""
    import cp_openings
    import robot_cp
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    per = [cp_openings.connection_points(
        V, F, q.argmax(-1), min_v=min_v, classes=(CE, CT), dedupe_mm=dedupe_mm,
        probs=q, vertex_conf=vc, ct_depth_min_mm=1.0, cluster_mm=cluster_mm,
        step_path=step_path) for q in pbs]
    if len(per) == 1:
        cps = per[0]
        for c in cps:
            c["_votes"] = 1
    else:
        cps = robot_cp._vote2(per, cluster_mm=oy_mm, min_votes=1)
    return cps


def kahin(P, G, Gd, diag):
    """MUKEMMEL gate: havuzda toleransta aday olan her GT bir TP (one-to-one, Macar)."""
    from sina_kume import esle_macar
    D = np.tile([0.0, 0, 1], (len(P), 1)) if len(P) else np.zeros((0, 3))
    Gd2 = Gd
    tp, _fp, _fn, _b = esle_macar(P, D, G, Gd2, diag, 0.0, 180.0, True)
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinir", type=int, default=0)
    a = ap.parse_args()
    import protokol
    protokol.tez_dogrula()
    from korpus_kimlik import step_kimlik as SK
    from sina_kume import f1w

    sv = d6_kayit.sinav()
    kayit = d6_kayit.yukle(set(sv["pidler"]))
    S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
    dosyalar = sorted(glob.glob(os.path.join(ONBELLEK, "*.npz")))
    if a.sinir:
        dosyalar = dosyalar[:a.sinir]
    print(f"onbellekte {len(dosyalar)} parca\n")

    # (cluster, dedupe, oy) uclulerI -- mevcut urun ilk sirada
    AYAR = [(3.0, 10.0, 5.0)]                      # MEVCUT
    for de in (3.0, 2.0, 1.0, 0.0):
        AYAR.append((3.0, de, 5.0))
    for cl in (2.0, 1.0, 0.0):
        AYAR.append((cl, 10.0, 5.0))
    for oy in (3.0, 2.0, 1.0, 0.0):
        AYAR.append((3.0, 10.0, oy))
    AYAR.append((1.0, 2.0, 2.0))                   # ucu birden dar
    AYAR.append((2.0, 3.0, 3.0))

    sonuc = {}
    taban = None
    print(f"{'cl/de/oy':<14}{'KAHIN':>8}{'dusuk':>8}{'cok':>8}{'aday':>8}"
          f"{'sure_s':>8}{'fark':>9}")
    for (cl, de, oy) in AYAR:
        t0 = time.time()
        satir, n_aday = [], 0
        for f in dosyalar:
            pid = os.path.splitext(os.path.basename(f))[0]
            r = kayit.get(pid)
            if r is None or pid not in S:
                continue
            d = np.load(f)
            V = np.ascontiguousarray(d["V"], np.float64)
            Fq = np.ascontiguousarray(d["F"], np.int64)
            pbs = [np.asarray(q, float) for q in d["pbs"]]
            try:
                cps = havuz(V, Fq, pbs, S[pid], cl, de, oy)
            except Exception:
                continue
            P = np.array([c["point"] for c in cps], float) if cps else np.zeros((0, 3))
            n_aday += len(P)
            G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
            rj = "cok" if r["n"] >= 8 else "dusuk"
            tp = kahin(P, G, Gd, r["diag"]) if len(G) else 0
            satir.append((rj, tp, 0, len(G) - tp))
        v = f1w(satir)
        dl = f1w([s for s in satir if s[0] == "dusuk"])
        ck = f1w([s for s in satir if s[0] == "cok"])
        sure = time.time() - t0
        if taban is None:
            taban = (v, dl, ck)
        fark = f"{v-taban[0]:+.4f}" if taban else ""
        print(f"{cl:.0f}/{de:.0f}/{oy:.0f}".ljust(14) +
              f"{v:>8.4f}{dl:>8.4f}{ck:>8.4f}{n_aday:>8}{sure:>8.0f}{fark:>9}")
        sonuc[f"{cl}/{de}/{oy}"] = {"kahin": v, "dusuk": dl, "cok": ck,
                                    "aday": n_aday, "sure_s": sure}

    en = max((k for k in sonuc), key=lambda k: sonuc[k]["kahin"])
    t = sonuc[f"3.0/10.0/5.0"]
    e = sonuc[en]
    print(f"\nMEVCUT 3/10/5 : kahin {t['kahin']:.4f} (dusuk {t['dusuk']:.4f} "
          f"cok {t['cok']:.4f}) aday {t['aday']}")
    print(f"EN IYI  {en:<10}: kahin {e['kahin']:.4f} (dusuk {e['dusuk']:.4f} "
          f"cok {e['cok']:.4f}) aday {e['aday']}")
    go = (e["kahin"] - t["kahin"] >= 0.01 and e["cok"] - t["cok"] >= 0.015
          and t["dusuk"] - e["dusuk"] <= 0.005)
    print(f"\nGO OLCUTU (genel +0.01 / cok-CP +0.015 / dusuk-CP kayip <=0.005): "
          f"{'GECTI' if go else 'KALDI'}")
    print(f"  genel {e['kahin']-t['kahin']:+.4f} | cok-CP {e['cok']-t['cok']:+.4f} | "
          f"dusuk-CP {e['dusuk']-t['dusuk']:+.4f} | aday artisi "
          f"x{e['aday']/max(t['aday'],1):.2f}")
    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump({"sonuc": sonuc, "mevcut": "3.0/10.0/5.0", "en_iyi": en, "go": go},
                  f, indent=1, ensure_ascii=False)
    print(f"makbuz -> {MAKBUZ}")


if __name__ == "__main__":
    main()
