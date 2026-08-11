# -*- coding: utf-8 -*-
"""GEOMETRI ANAHTARI: ikiz parcalari saptayan imza -- URETICI BETIK (eksikti).

NEDEN YAZILDI: `results/_strict_geometry_keys.json` dosyasini **10 betik OKUYOR, hicbiri
YAZMIYOR** -- `_der_tam.pkl`'in provenance bosluguyla ayni durum (turet.py o bosluk icin
yazilmisti). Anahtar 1963 parcayi kapsiyor; korpus 4720'ye ciktigi icin **2757 parcanin
anahtari YOK**.

BU NEDEN KRITIK: [[geometry-twin-leakage]] -- parcalarin %80'inin korpusta ikizi var.
Sizinti korumalari ve `protokol.egitim_maskesi` GRUP uzerinden calisir; anahtari olmayan
parca "kendi basina grup" sayilir. Yani yeni veride OLCUM KUMESININ IKIZI varsa egitime
girer ve manset SESSIZCE SISER. Turetmeden ONCE kapatilmali.

ESKI FORMAT COZULDU ama BIREBIR taklit EDILMIYOR:
    [8.1, 50.4, 71.8]|v12|f13|c19|p161|h5.2.10.2.0.0.0.0.0.0.0.0
    v = ceil(log2(tepe)) , f = ceil(log2(yuz)) , basta SIRALI sinir kutusu, sonda histogram
Sinir kutusunda 0.1mm'lik sapma var (8.2 -> 8.1), yani eski anahtar biraz FARKLI bir mesh
surumunden uretilmis. Taklit etmek yerine YENI anahtar yazilir ve **eski GRUPLAMAYI
yeniden uretip uretmedigi SINANIR** (dogrulama asagida).

TASARIM: anahtar remesh gurultusune DAYANIKLI olmali ([[robot-nondeterminism]]: pymeshlab
surecler arasi ~0.4mm oynuyor) ama farkli parcalari AYIRMALI.
"""
import hashlib
import io
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BBOX_KOVA = 0.5      # mm -- remesh gurultusu ~0.4mm, kova ondan buyuk olmali
HACIM_KOVA = 0.02    # doluluk orani kovasi


def anahtar(V, F):
    """Rotasyon-bagimsiz, remesh-gurultusune dayanikli geometri imzasi."""
    V = np.asarray(V, float)
    bb = np.sort(V.max(0) - V.min(0))
    bbk = [round(float(x) / BBOX_KOVA) for x in bb]
    nv, nf = len(V), len(np.asarray(F))
    # DOLULUK: mesh hacmi / sinir kutusu hacmi -- ayni dis olcude farkli ic yapiyi ayirir
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=V, faces=np.asarray(F, np.int64), process=False)
        hac = abs(float(m.volume)) / max(float(np.prod(bb)), 1e-9)
    except Exception:
        hac = -1.0
    hk = round(hac / HACIM_KOVA) if hac >= 0 else -1
    # MERKEZDEN UZAKLIK HISTOGRAMI (rotasyon-bagimsiz).
    # KABALASTIRILDI (ilk surum 5/5 kacirilan ikizde SUCLUYDU): 12 kova + %1 yuvarlama
    # tek kovada 1 PUANLIK tesselasyon gurultusuyle anahtari degistiriyordu
    # (h0.4.7... vs h0.5.7...). 8 kova + %5 adim, gurultuye dayanikli.
    c = V.mean(0)
    r = np.linalg.norm(V - c, axis=1)
    r = r / max(r.max(), 1e-9)
    h, _ = np.histogram(r, bins=8, range=(0, 1))
    h = (np.round(100 * h / max(len(r), 1) / 5) * 5).astype(int)
    # TEPE/YUZ SAYISI ANAHTARDAN CIKARILDI: mesh yogunlugu TESSELASYON artefaktidir,
    # geometri degil. Bir vakada ayni parca `v12` ve `v13` aliyordu (2'nin kuvveti sinirini
    # gecmis). bbox + doluluk + histogram ayirt etmeye yetiyor (dogrulama asagida).
    return f"b{bbk[0]}.{bbk[1]}.{bbk[2]}|d{hk}|h" + ".".join(map(str, h))


def dogrula(n=300, tohum=0):
    """YENI anahtar, ESKI gruplamayi yeniden uretiyor mu? (eski parcalarda sinanir)"""
    import glob
    from infer_step_cp import step_to_mesh
    from korpus_kimlik import step_kimlik
    eski = json.load(io.open("results/_strict_geometry_keys.json", encoding="utf-8"))
    sm = {step_kimlik(s): s for s in glob.glob("all_wscad_stp/*.stp")}
    ortak = [p for p in eski if p in sm]
    rng = np.random.RandomState(tohum)
    sec = [ortak[i] for i in rng.permutation(len(ortak))[:n]]
    yeni = {}
    for i, p in enumerate(sec, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(sec)}", flush=True)
        try:
            V, F = step_to_mesh(sm[p])
            yeni[p] = anahtar(V, F)
        except Exception:
            pass
    ok = [p for p in sec if p in yeni]
    # CIFT bazinda uyum: ayni eski grupta olanlar yeni anahtarda da ayni mi?
    ee = ey = ye = 0
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            a, b = ok[i], ok[j]
            e = eski[a] == eski[b]
            y = yeni[a] == yeni[b]
            if e and y:
                ee += 1
            elif e and not y:
                ey += 1
            elif y and not e:
                ye += 1
    print(f"\nDOGRULAMA ({len(ok)} parca, {len(ok)*(len(ok)-1)//2} cift):")
    print(f"  eski AYNI grup & yeni AYNI  : {ee}   (korunan ikizler)")
    print(f"  eski AYNI grup & yeni FARKLI: {ey}   <- IKIZ KACIRILDI (kotu)")
    print(f"  eski FARKLI & yeni AYNI     : {ye}   <- YANLIS BIRLESTIRME (kotu)")
    duy = ee / max(ee + ey, 1)
    kes = ee / max(ee + ye, 1)
    print(f"  duyarlilik {duy:.4f} | kesinlik {kes:.4f}")
    gecti = duy >= 0.95 and kes >= 0.95
    print(f"  KILL: ikisi de >=0.95 -> {'GECTI' if gecti else 'GECMEDI'}")
    return gecti, {"ee": ee, "ey": ey, "ye": ye, "duyarlilik": duy, "kesinlik": kes}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogrula", type=int, default=300)
    a = ap.parse_args()
    g, r = dogrula(a.dogrula)
    with io.open("results/geometri_anahtar_dogrulama.json", "w", encoding="utf-8") as f:
        json.dump({"gecti": bool(g), **r, "bbox_kova": BBOX_KOVA}, f, indent=1)
    print("makbuz -> results/geometri_anahtar_dogrulama.json")
