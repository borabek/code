# -*- coding: utf-8 -*-
"""ES-EKSENLI AILE / DIZI DUZENLILIGI: "bu aday yalniz mi, yoksa bir dizinin uyesi mi?"

FIZIK. Klemens govdesinde tel girisleri TEK BASINA durmaz: ayni yuzde bir SIRA
olustururlar (dik dizi) ve govdenin arkasinda ayni EKSEN uzerinde karsilik gelen
bir ikinci acikliga (vida yuvasi / kablo kanali) bakarlar. Sahte adaylar --
plastik kenari, vida basi, model gurultusu -- bu duzene uymaz.

KAPALI KOL DEGIL. Hafizadaki "topoloji yaricapi kapandi (R=6.0)" kolu adayin
KOMSU SAYISINI bir yaricapta sayiyordu; buradaki olcu adayin KENDI YONUNE gore
hizalanmis yapisal iliskidir (eksen boyu ortak, eksene dik dizi, dizi araligi
duzenliligi). Ayni bilgi degil: komsu sayisi yonden bagimsizdir, bu degildir.

Sutunlar (6):
  es_eksen_n   aday eksenine (1.5mm boru icinde) dusen DIGER aday sayisi
  dik_dizi_n   adayin agiz duzleminde (|t|<1mm) yatan diger aday sayisi
  aile_orani   dik_dizi_n / toplam aday  (olcek bagimsiz)
  aralik_cv    dik dizi komsu araliklarinin degisim katsayisi (KUCUK = duzenli)
  disa_bakis   (agirlik merkezi -> aday) . yon   [+ = disari bakiyor]
  en_yakin_dik dik dizideki en yakin komsunun mesafesi / kosegen
"""
import numpy as np

OZ_AD = ["es_eksen_n", "dik_dizi_n", "aile_orani", "aralik_cv",
         "disa_bakis", "en_yakin_dik"]

BORU_R = 1.5      # es-eksen borusu yaricapi (mm)
DUZLEM_T = 1.0    # agiz duzlemi kalinligi (mm)
EN_AZ = 0.5       # kendini saymamak icin alt esik (mm)


def _birim(V):
    V = np.asarray(V, float).reshape(-1, 3)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def oznitelik(P, D, Pu, diag):
    """(n, 6). `P`,`D` secenek konum/yonu; `Pu` parcanin TUM aday konumlari."""
    P = np.asarray(P, float).reshape(-1, 3)
    D = _birim(D)
    Pu = np.asarray(Pu, float).reshape(-1, 3)
    n = len(P)
    X = np.zeros((n, len(OZ_AD)))
    if not n or len(Pu) < 2:
        return X
    diag = max(float(diag), 1e-6)
    merkez = Pu.mean(0)
    # Cok buyuk havuzlarda referans kumesini seyrelt: maliyet n x m.
    if len(Pu) > 4000:
        Pu = Pu[np.random.default_rng(0).choice(len(Pu), 4000, replace=False)]
    m = len(Pu)
    for i in range(n):
        v = Pu - P[i]                       # (m,3)
        t = v @ D[i]                        # eksen boyu koordinat
        dik = np.linalg.norm(v - t[:, None] * D[i][None, :], axis=1)
        es_eksen = (dik < BORU_R) & (np.abs(t) > EN_AZ)
        duzlem = (np.abs(t) < DUZLEM_T) & (dik > EN_AZ)
        X[i, 0] = es_eksen.sum()
        X[i, 1] = duzlem.sum()
        X[i, 2] = duzlem.sum() / m
        dd = np.sort(dik[duzlem])
        if len(dd) >= 3:
            ara = np.diff(dd)
            ara = ara[ara > 1e-6]
            if len(ara) >= 2:
                X[i, 3] = float(ara.std() / max(ara.mean(), 1e-9))
        X[i, 4] = float((P[i] - merkez) @ D[i]) / diag
        X[i, 5] = (float(dd[0]) / diag) if len(dd) else 1.0
    return X
