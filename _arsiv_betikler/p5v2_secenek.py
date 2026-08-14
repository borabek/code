# -*- coding: utf-8 -*-
"""p5-v2 SECENEK URETICI -- gate'ten ONCE, ORTAK secim icin.

OLCULDU (`gate-once-poz-sonra-tavani-kirpiyor`): gate'i poz seciminden ONCE
uygulamak ortak kahin tavanini 0.3781 -> 0.3155 kirpiyor. Bu modul, HAM aday
havuzu icin secenekleri uretir; secim ve gate SONRA gelir.

SECENEK TURLERI (tez sadakati: MEVCUT her zaman 0 numarali secenek):
  0 MEVCUT   : `v_o` -- tezin kendi turetmesi, GERCEK fallback
  1 SILINDIR : B-rep silindir agzi (+/- eksen)
  2 PLANAR   : duzlemsel yuz ic halkasi (slot/kare giris) (+/- normal)
  3 NULL     : adayi AT (p5-v2 bir adayi elemekte serbest)

Bir FIZIKSEL AGIZ en fazla bir adaya verilir -- bu kisit secici tarafinda
(bipartite) uygulanir; burada yalniz secenekler ve OZNITELIKLERI uretilir.
"""
import numpy as np

import os
MM_MAX = 8.0            # agiz-aday mesafe siniri
# ABLASYON: planar (duzlemsel ic halka) seceneklerini KAPAT -> kazancin
# silindirlerden mi planarlardan mi geldigini ayirir.
P5V2_PLANAR = os.environ.get("P5V2_PLANAR", "1") == "1"
TUR_MEVCUT, TUR_SILINDIR, TUR_PLANAR, TUR_NULL = 0, 1, 2, 3
OZ_AD = ["tur_silindir", "tur_planar", "tur_null",
         "mesafe", "mesafe_norm", "aci_mevcut", "yaricap", "uzunluk",
         "esd_r", "cevre", "alan", "eksen_hiza", "komsu_uyum",
         "gate_skoru", "votes", "n_aday", "n_secenek"]
# PARCA-ICI GORELI OZNITELIKLER: DENENDI, GERI ALINDI (2026-08-09).
# Gate tarihindeki en buyuk tek kazanc parca-ici z-skordu
# ([[parca-ici-zskor-dagitildi]]) ve ayni fikri buraya tasidim:
# aday-ici siralama (mesafe/aci/yaricap) + parca genelinde z-skor, 6 sutun.
# OLCULDU (LOMO, ayni kurulum, tek degisken):
#     ham 23 sutun -> secim +0.0391 | uctan uca 0.1465
#     ham 17 sutun -> secim +0.0501 | uctan uca 0.1649   <- IYI OLAN
# BOZDU. Muhtemel sebep: gate'in oznitelikleri markalar arasi kiyaslanamaz
# MUTLAK buyukluklerdi; p5-v2'ninkiler ZATEN goreli (bu adaydan bu agza mesafe).
# Siralama bilgi katmiyor, RF'nin asiri uyacagi 6 sutun gurultu katiyor.
# TEKRAR DENEME -- once oznitelik uzayini degistir, sonra normalizasyonu.
# NOT: `agiz_kimlik` OZ_AD'de YOK -- o bir OZNITELIK degil, bipartite KISIT
# anahtaridir ve tuple'in 4. ogesi olarak ayri doner. (Ilk surumde listeye
# koymustum: 18 ad / 17 deger uyusmazligi olurdu.)


def _birim(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def secenekler(P, D, cyls, acik, diag, gate_s=None, votes=None, komsu=None):
    """Her aday icin secenek listesi. Doner: liste[liste[(konum, yon, oz, agiz_id)]].

    `agiz_id`: ayni fiziksel agzi kullanan secenekler AYNI kimligi tasir; secici
    bir agzi iki adaya veremesin diye. MEVCUT ve NULL icin -1 (kisit disi).
    """
    P = np.asarray(P, float); D = np.asarray(D, float)
    n = len(P)
    gate_s = np.zeros(n) if gate_s is None else np.asarray(gate_s, float)
    votes = np.zeros(n) if votes is None else np.asarray(votes, float)
    # fiziksel agizlar: (konum, yon, tur, yaricap, uzunluk, esd_r, cevre, alan)
    agizlar = []
    for c in (cyls or []):
        a = _birim(c["axis"])
        uzn = float(np.linalg.norm(np.asarray(c["mouth_b"], float) -
                                   np.asarray(c["mouth_a"], float)))
        for m in (c["mouth_a"], c["mouth_b"]):
            agizlar.append((np.asarray(m, float), a, TUR_SILINDIR,
                            float(c["radius"]), uzn, 0.0, 0.0, 0.0))
    for o in ((acik or []) if P5V2_PLANAR else []):
        nrm = _birim(o.get("normal", [0, 0, 1]))
        agizlar.append((np.asarray(o["center"], float), nrm, TUR_PLANAR,
                        0.0, 0.0, float(o.get("esd_r", 0.0)),
                        float(o.get("cevre", 0.0)), float(o.get("alan", 0.0))))
    out = []
    for i in range(n):
        o = []
        u0 = _birim(D[i])

        def oz(mes, aci, yar, uzn, esd, cev, aln, tur, yon, n_sec):
            return [float(tur == TUR_SILINDIR), float(tur == TUR_PLANAR),
                    float(tur == TUR_NULL),
                    mes, mes / max(diag, 1e-6), aci, yar, uzn, esd, cev, aln,
                    float(np.max(np.abs(yon))),
                    abs(float(yon @ komsu)) if komsu is not None else 0.0,
                    float(gate_s[i]), float(votes[i]), float(n), float(n_sec)]

        # 0) MEVCUT -- tezin cevabi, HER ZAMAN ilk
        o.append((P[i], u0, oz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               TUR_MEVCUT, u0, 0), -1))
        for k, (m, a, tur, yar, uzn, esd, cev, aln) in enumerate(agizlar):
            mes = float(np.linalg.norm(m - P[i]))
            if mes > MM_MAX:
                continue
            for sg in (1.0, -1.0):
                y = sg * a
                aci = float(np.degrees(np.arccos(np.clip(abs(float(y @ u0)), -1, 1))))
                o.append((m, y, oz(mes, aci, yar, uzn, esd, cev, aln, tur, y, 0), k))
        # 3) NULL -- adayi at
        o.append((None, None, oz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 TUR_NULL, u0, 0), -1))
        for t in o:                       # n_secenek'i geriye yaz
            t[2][-1] = float(len(o))
        out.append(o)
    return out
