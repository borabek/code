# -*- coding: utf-8 -*-
"""PERIYODIK YAPI: klemens kontaklari sabit OTELEME VEKTORLERIYLE tekrarlanir.

TESHIS (2026-08-11, D6-ici LOMO): yon secimi cozuldu (kahin farki yalniz
+0.0057) ama recall 0.177 iken havuz tavani 0.525 -- havuzda aday VAR, siralayici
onu AYIRT EDEMIYOR. Aday-basina bilgi tukendi; kullanilmamis bilgi PARCA
DUZEYINDE.

OLCULDU (D6, >=6 CP olan 1096 parca, 16968 GT): parcanin EN SIK OTELEME
VEKTORUYLE baska bir GT'ye ulasan GT orani **%90.8** (parca medyani %100).
Ornek adimlar: 3.5 / 6.0 / 9.0 / 10.0 / 14.25 mm. Yapi gercek ve cok guclu.

ILK DENEMEM YANLISTI: kontaklari 1B SIRA (dogru + adim) olarak modelledim,
GT'nin yalniz %5'i uydu. Klemens 2B duzen (kolon x kat) oldugu icin bir dogruya
izdusum satirlari ust uste bindiriyor. Dogru ilkel DOGRU degil OTELEME VEKTORU.

NEDEN ONCEKI PERIYODIKLIK KOLU (K2.1) OLMUSTU: yayilan noktaya KAYNAK NOKTANIN
YONU kopyalaniyordu -> tespit +0.0126 ama robot -0.0100. Burada yon
KOPYALANMIYOR. Periyodiklik yalnizca bir OZNITELIKTIR; yonu her zaman yon
bankasi + ortak siralayici secer.

TEZ: turetme, remesh, `v_o` degismez. Bu bir puanlama olcusudur.
"""
import collections

import numpy as np

IZGARA = 0.25          # oteleme vektoru yuvarlama (mm)
MIN_UZ, MAKS_UZ = 2.0, 40.0
EN_COK_T = 3           # kac oteleme vektoru tutulur
K_MAKS = 3             # kac adim ileri/geri yayilir
MIN_DESTEK = 2         # bir otelemenin gecerli sayilmasi icin en az bu kadar cift
OZ_AD = ["per_var", "per_mesafe", "per_yanal", "per_destek", "per_adim",
         "per_k", "per_yon_uyum", "per_tohum_n"]


def _birim(V):
    V = np.asarray(V, float).reshape(-1, 3)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def otelemeler(P, en_cok=EN_COK_T):
    """Tohum konumlarindan EN SIK oteleme vektorleri. Doner: liste[(t, destek)].

    +t ve -t AYNI otelemedir; kanonik isaret sozlukte birlestirilir.
    """
    P = np.asarray(P, float).reshape(-1, 3)
    if len(P) < 2:
        return []
    df = (P[:, None, :] - P[None, :, :]).reshape(-1, 3)
    uz = np.linalg.norm(df, axis=1)
    df = df[(uz >= MIN_UZ) & (uz <= MAKS_UZ)]
    if not len(df):
        return []
    say = collections.Counter()
    for v in df:
        k = tuple(np.round(v / IZGARA).astype(int))
        if k < tuple(-x for x in k):
            k = tuple(-x for x in k)
        say[k] += 1
    out = []
    for k, n in say.most_common(en_cok * 4):
        if n < MIN_DESTEK:
            break
        t = np.asarray(k, float) * IZGARA
        if any(np.linalg.norm(t - s) < 0.5 or np.linalg.norm(t + s) < 0.5
               for s, _ in out):
            continue                      # zaten yakin bir oteleme var
        out.append((t, int(n)))
        if len(out) >= en_cok:
            break
    return out


def ongorulen(P_tohum, D_tohum, tler, k_maks=K_MAKS):
    """Tohum + k*t ile ONGORULEN konumlar. Doner: (S, Sd, destek, adim, kk).

    `Sd` tohumun yonudur ve YALNIZCA OZNITELIK icin tasinir -- ongorulen konuma
    yon ATAMAZ. (K2.1 tam burada hata yapmisti.)
    """
    P = np.asarray(P_tohum, float).reshape(-1, 3)
    D = _birim(D_tohum)
    # ARA ADIMLAR (k = 0.5, 1.5, ...) ZORUNLU. Birim test: 6mm adimli bir sirada
    # tohumlar bir atlayarak dususe, en sik oteleme 12mm cikar ve ARADAKI
    # kontaklar hicbir zaman ongorulmez -- yani ozniteligin en cok gerektigi
    # durumda tam olarak susar. Yarim adimlar bu bosluga bakar; yanlis olduklari
    # yerde modele yalnizca "uzak" bilgisi verirler.
    adimlar = [k / 2.0 for k in range(1, 2 * k_maks + 1)]
    S, Sd, ds, ad, kk = [], [], [], [], []
    for t, n in tler:
        for k in adimlar:
            for sg in (1.0, -1.0):
                S.append(P + sg * k * t)
                Sd.append(D)
                ds.append(np.full(len(P), n, float))
                ad.append(np.full(len(P), float(np.linalg.norm(t))))
                kk.append(np.full(len(P), float(k)))
    if not S:
        z = np.zeros((0, 3))
        return z, z, np.zeros(0), np.zeros(0), np.zeros(0)
    return (np.vstack(S), np.vstack(Sd), np.concatenate(ds),
            np.concatenate(ad), np.concatenate(kk))


def oznitelik(P, D, P_tohum, D_tohum):
    """Her (konum, yon) icin periyodik yapi olculeri. Doner: (n, len(OZ_AD))."""
    P = np.asarray(P, float).reshape(-1, 3)
    n = len(P)
    X = np.zeros((n, len(OZ_AD)))
    if not n:
        return X
    X[:, 1] = X[:, 2] = 1e3               # uzak varsayilan
    P_tohum = np.asarray(P_tohum, float).reshape(-1, 3)
    if len(P_tohum) < 2:
        return X
    tler = otelemeler(P_tohum)
    if not tler:
        return X
    S, Sd, ds, ad, kk = ongorulen(P_tohum, D_tohum, tler)
    if not len(S):
        return X
    D = _birim(D)
    # her aday icin EN YAKIN ongorulen konum
    d = np.linalg.norm(P[:, None, :] - S[None, :, :], axis=-1)
    j = np.argmin(d, axis=1)
    m = d[np.arange(n), j]
    X[:, 0] = 1.0
    X[:, 1] = m
    # yanal: ongorulen konumun YONUNE dik bilesen (eksenel kayma daha zararsiz)
    w = P - S[j]
    al = (w * Sd[j]).sum(1)
    X[:, 2] = np.linalg.norm(w - al[:, None] * Sd[j], axis=1)
    X[:, 3] = ds[j]
    X[:, 4] = ad[j]
    X[:, 5] = kk[j]
    X[:, 6] = np.abs((D * Sd[j]).sum(1))
    X[:, 7] = float(len(P_tohum))
    return X
