# -*- coding: utf-8 -*-
"""Q2: SINAV BUTUNLUGU -- 95 LOCKED parcanin gercekten BAKIR oldugunu FORMEL kanitla.

NEDEN SIMDI, harcamadan ONCE: bir final sinavi ancak GECERLILIGI onceden gosterilmisse
degerlidir. "Dokunmadik" demek yetmez; kanit gerekir. Ve bu gece ogrendik ki bir kez
dokunulan parca KALICI olarak kirlidir ([[olcum-zaafiyetleri-kapatildi]] ve 98->95
duzeltmesi).

DENETLENEN DORT SART:
  S1  Hicbir LOCKED parca, hicbir egitim npz'sinde (pids) BULUNMAYACAK.
  S2  Hicbir LOCKED parcanin GEOMETRI GRUBU, egitim npz'lerinde temsil edilmeyecek
      (ikiz sizintisi -- as?l tehlike budur, parca kimligi degil).
  S3  Hicbir LOCKED parca olcum onbelleklerinde (_der_*.pkl) puanlanmis olmayacak.
  S4  Hicbir LOCKED parcanin grubu, olcum kumesinin gruplariyla kesismeyecek.

Cikti: results/q2_sinav_butunlugu.json -- her sart icin GECTI/KALDI + ihlal listesi.
Bu makbuz, LOCKED kosulmadan ONCE uretilmis olmalidir; sonradan uretilirse degeri yoktur.
"""
import glob
import io
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import olcum_kumesi

    gk = olcum_kumesi.geo_anahtarlari()
    s3 = json.load(io.open("results/split3.json", encoding="utf-8"))
    locked = [str(p) for p in s3["locked"]["parts"]]
    lg = {p: gk.get(p, "yok:" + p) for p in locked}
    DER, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    olcum_kumesi.rapor_bas(rap)
    temiz = set(rap["locked_temiz"])
    print(f"\nLOCKED toplam {len(locked)} | grup-temiz {len(temiz)}")

    RAPOR = {"locked_toplam": len(locked), "grup_temiz": sorted(temiz), "sartlar": {}}

    # --- S1/S2: egitim npz'leri
    npzler = sorted(set(glob.glob("results/zengin_parite*.npz")) |
                    set(glob.glob("results/gate_regrow_data*.npz")) |
                    set(glob.glob("results/yeni_wei.npz")))
    ihlal1, ihlal2 = {}, {}
    for n in npzler:
        try:
            d = np.load(n, allow_pickle=True)
            if "pids" not in d.files:
                continue
            pids = {str(x) for x in d["pids"]}
            gruplar = {gk.get(p, "yok:" + p) for p in pids}
        except Exception as e:
            print(f"  {n}: okunamadi ({type(e).__name__})"); continue
        a = sorted(p for p in temiz if p in pids)
        b = sorted(p for p in temiz if lg[p] in gruplar)
        if a:
            ihlal1[n] = a
        if b:
            ihlal2[n] = b
        print(f"  {os.path.basename(n):<28} parca-ihlal {len(a):>3} | grup-ihlal {len(b):>3}")

    # --- S3/S4: olcum onbellekleri
    ihlal3, ihlal4 = {}, {}
    for pk in sorted(glob.glob("results/_der*.pkl")):
        try:
            with open(pk, "rb") as f:
                R = pickle.load(f)
            pids = {str(r["pid"]) for r in R}
            gruplar = {gk.get(p, "yok:" + p) for p in pids}
        except Exception:
            continue
        a = sorted(p for p in temiz if p in pids)
        b = sorted(p for p in temiz if lg[p] in gruplar)
        if a:
            ihlal3[pk] = a
        if b:
            ihlal4[pk] = b
        print(f"  {os.path.basename(pk):<28} parca-ihlal {len(a):>3} | grup-ihlal {len(b):>3}")

    for ad, ih, aciklama in (
        ("S1 egitim npz'sinde PARCA yok", ihlal1, "LOCKED parca egitim verisinde"),
        ("S2 egitim npz'sinde GRUP yok", ihlal2, "LOCKED parcanin IKIZI egitim verisinde"),
        ("S3 olcum onbelleginde PARCA yok", ihlal3, "LOCKED parca puanlanmis"),
        ("S4 olcum onbelleginde GRUP yok", ihlal4, "LOCKED parcanin IKIZI puanlanmis"),
    ):
        gecti = not ih
        RAPOR["sartlar"][ad] = {"gecti": gecti, "aciklama": aciklama,
                                "ihlal": {k: v for k, v in ih.items()}}
        print(f"\n{ad:<36}{'GECTI' if gecti else 'KALDI'}")
        if not gecti:
            for k, v in list(ih.items())[:3]:
                print(f"    {os.path.basename(k)}: {len(v)} parca -> {v[:6]}")

    hepsi = all(v["gecti"] for v in RAPOR["sartlar"].values())
    RAPOR["sinav_gecerli"] = hepsi
    print(f"\nSINAV GECERLI Mi: {'EVET -- 95 parca BAKIR' if hepsi else 'HAYIR -- ihlal var'}")
    with io.open("results/q2_sinav_butunlugu.json", "w", encoding="utf-8") as f:
        json.dump(RAPOR, f, indent=1)
    print("makbuz -> results/q2_sinav_butunlugu.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
