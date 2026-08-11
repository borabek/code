# -*- coding: utf-8 -*-
"""Agiz tanimlayicilarini EGITIM korpusu + D7 icin uret ve diske yaz.

Olculdu (`results/agiz_ayirt_edicilik.json`, egitim yok, saf AUC): parca ICINDE
yaricap 0.690, es_eksen 0.293, narinlik 0.317, girme_kenar 0.331, girme 0.343
(0.29 ile 0.69 esit guclu, yon ters). Yani bir agzin gercek tel girisi mi
oldugunu ayirt eden bilgi VAR -- yalnizca 58 gate ozniteliginde yoktu.

Bu betik o tanimlayicilari kalici hale getirir. Yeniden baslatilabilir:
her parca kendi dosyasina yazilir, var olan atlanir.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import trimesh

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import agiz_tanimlayici as AT   # noqa: E402
import brep_havuz               # noqa: E402
import d6_kayit                 # noqa: E402
import kanonik_d7 as K          # noqa: E402

CIK = "results/_agiz_tan"
os.makedirs(CIK, exist_ok=True)

ISLER = [
    ("d7", "results/_p1_olasilik_d7", "results/_d7_silindirler.pkl",
     "results/_d7_acikliklar.pkl", None),
    ("d6", "results/_p1_olasilik_g7", "results/_d6_silindirler.pkl",
     "results/_d6_acikliklar.pkl", None),
    ("tam", "results/_p1_olasilik_brepegit", "results/_brepegit_silindirler.pkl",
     "results/_brepegit_acikliklar.pkl", "results/brep_egitim_kumesi.json"),
]


def kayitlar(ad, kume):
    if ad == "d6":
        return d6_kayit.yukle(set(d6_kayit.sinav()["pidler"]))
    if ad == "d7":
        return K.yukle(json.load(open("results/d7_sinav_kumesi.json"))["pidler"])
    return K.yukle(json.load(open(kume))["pidler"])


for ad, ob, cylf, acf, kume in ISLER:
    cy = pickle.load(open(cylf, "rb"))
    ac = pickle.load(open(acf, "rb"))
    kay = kayitlar(ad, kume)
    t0 = time.time()
    yazilan = atlanan = hatali = 0
    for i, (pid, r) in enumerate(sorted(kay.items())):
        yol = f"{CIK}/{ad}_{pid}.npz"
        if os.path.exists(yol):
            atlanan += 1
            continue
        f = f"{ob}/{pid}.npz"
        if not os.path.exists(f):
            hatali += 1
            continue
        P, D, met = brep_havuz.brep_adaylari(cy.get(pid), ac.get(pid), meta=True)
        if not len(P):
            np.savez_compressed(yol, T=np.zeros((0, len(AT.AD))))
            yazilan += 1
            continue
        z = np.load(f)
        mesh = trimesh.Trimesh(np.asarray(z["V"], float),
                               np.asarray(z["F"], np.int64), process=False)
        T = AT.tanimla(P, D, met, mesh, float(r["diag"]))
        np.savez_compressed(yol, T=T.astype(np.float32))
        yazilan += 1
        if yazilan % 100 == 0:
            print(f"  {ad} {i+1}/{len(kay)} yazilan={yazilan} "
                  f"{(time.time()-t0)/max(yazilan,1):.2f}s/parca", flush=True)
    print(f"{ad} BITTI: yazilan {yazilan} | onbellekte {atlanan} | "
          f"olasiligi yok {hatali}", flush=True)
print("hepsi bitti ->", CIK)
