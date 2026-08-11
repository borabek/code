# -*- coding: utf-8 -*-
"""SIRA DAMGALAMA: iki-uc capadan TUM SIRAYI bas.

GEREKCE. Yogun klemenste (NIT 24 CP/parca) aday-basina siralama 12x'te
tikaniyor: 4467 secenek icinde 24 dogruyu uste tasimak icin ~50-100x gerek.
Ama olculdu ki GT'lerin **%90.8'i** parcanin en sik OTELEME VEKTORUYLE baska
bir GT'ye ulasiyor. Yani kontaklarin yeri BIRBIRINDEN cikarilabilir.

Aday-basina karar yerine SIRA-DUZEYI karar: birkac YUKSEK GUVENLI capa bul,
onlardan oteleme vektorunu cikar, sirayi iki yone dogru uzat ve her ongorulen
noktayi UCUZ FIZIK TESTIYLE dogrula (agiz orada gercekten var mi). Kabul
edilen noktanin YONU capadan KOPYALANMAZ -- yon her zaman yon bankasindan
secilir (K2.1 tam burada hata yapmisti: tespit +0.0126 ama robot -0.0100).

Bu modul bir ONERI URETICIDIR; ciktisi mevcut secimin UZERINE eklenir ve
kabul/ret yine skor + NMS kurallarindan gecer.
"""
import numpy as np

import kafes

MIN_CAPA = 2           # oteleme cikarmak icin en az bu kadar capa
MAKS_ADIM = 12         # sirayi kac adim uzat (her iki yone)
TOL_MM = 1.0           # ongorulen nokta ile havuz adayi arasi kabul mesafesi


def _birim(V):
    V = np.asarray(V, float).reshape(-1, 3)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def oneriler(P_capa, D_capa, P_havuz, maks_adim=MAKS_ADIM, tol=TOL_MM):
    """Capalardan sirayi uzat, HAVUZDA KARSILIGI OLAN ongorulen noktalari don.

    Doner: (idx_havuz, kaynak_capa, adim) -- `idx_havuz` havuzdaki aday
    indeksleridir. YENI KONUM URETILMEZ: yalnizca havuzda ZATEN VAR OLAN ama
    dusuk skorlu adaylar "sira uzerinde" diye ONE CIKARILIR. Boylece konum
    dogrulugu havuzun garantisiyle sinirli kalir ve uydurma nokta olusmaz.
    """
    P_capa = np.asarray(P_capa, float).reshape(-1, 3)
    P_havuz = np.asarray(P_havuz, float).reshape(-1, 3)
    if len(P_capa) < MIN_CAPA or not len(P_havuz):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    tler = kafes.otelemeler(P_capa)
    if not tler:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    idx, kayn, adim = [], [], []
    gorulen = set()
    # ALT ADIMLAR: 1/2 VE 1/3. Birim test gosterdi ki capalar 18mm arayla
    # dususe en sik oteleme 18mm cikiyor; yarim adim 9mm verir ama gercek adim
    # 6mm'dir ve aradaki kontaklar HIC ongorulmez. Ucte-bir adim bu bosluga
    # bakar. (`kafes.ongorulen` de ayni duzeltmeyi tasir.)
    kesir = sorted({x / b for b in (1, 2, 3)
                    for x in range(1, b * maks_adim + 1)})
    for t, _n in tler:
        for k in kesir:
            for sg in (1.0, -1.0):
                S = P_capa + sg * k * t
                d = np.linalg.norm(P_havuz[:, None, :] - S[None, :, :], axis=-1)
                j = np.argmin(d, axis=1)
                m = d[np.arange(len(P_havuz)), j]
                for i in np.where(m <= tol)[0]:
                    if i in gorulen:
                        continue
                    gorulen.add(i)
                    idx.append(int(i))
                    kayn.append(int(j[i]))
                    adim.append(float(sg * k))
    return (np.asarray(idx, int), np.asarray(kayn, int),
            np.asarray(adim, float))


def oznitelik(P_havuz, D_havuz, P_capa, D_capa, **kw):
    """Havuzdaki her aday icin (sira_uzerinde, adim, capa_yon_uyumu).

    `kafes.oznitelik` bir MESAFE olcusu verir; bu ise KABUL EDILMIS bir sira
    uyeligi bayragidir -- ikisi farkli sorulari yanitlar ve birlikte kullanilir.
    """
    n = len(np.asarray(P_havuz, float).reshape(-1, 3))
    X = np.zeros((n, 3))
    i, c, a = oneriler(P_capa, D_capa, P_havuz, **kw)
    if not len(i):
        return X
    X[i, 0] = 1.0
    X[i, 1] = np.abs(a)
    Dh = _birim(D_havuz)
    Dc = _birim(D_capa)
    X[i, 2] = np.abs((Dh[i] * Dc[c]).sum(1))
    return X
