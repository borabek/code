# -*- coding: utf-8 -*-
"""B-rep GENISLETILMIS ADAY HAVUZU -- tek kaynak.

NEDEN: olculdu (`results/havuz_recall_d7.json`) ki D7 marka-disi havuz recall'u
segmentasyon tek basina **0.6654**. Donusum ~%48 oldugundan robot tavani ~0.32;
yani MEVCUT HAVUZLA 0.50 IMKANSIZ. Gate/secici/adet kollarinin hepsi bu tavanin
altindaydi ve hepsi kapandi. Baglayici kisit HAVUZ.

B-rep silindir agizlari + duzlemsel aciklik merkezleri EK aday kaynagi olarak
eklenince (D7, `results/brep_filtre_taramasi.json`):
    yalniz seg        recall 0.6654   13.5 aday/parca
    + B-rep (ham)     recall 0.8465  379.9
    + B-rep (dedupe3) recall 0.7852   98.2   <-- DIZ
CWT 0.3595 -> 0.6579, KLM 0.5663 -> 0.8313 (tabani ceken markalar).

DURUSTLUK: bu TEZ TURETMESI DEGIL. Tezin `v_o` agiz-ortasi turetmesi, 5 sinif
segmentasyon ve ~6000 remesh AYNEN durur; B-rep onerileri ONLARIN YANINA eklenen
ikinci bir aday kaynagidir ve `kaynak` alaniyla isaretlenir. Sonuclar "tez
sonucu" olarak DEGIL, "tez-omurgali geometrik genisletme" olarak raporlanir.
"""
import os

import numpy as np

DEDUPE_MM = 3.0


def _dedupe(P, mm):
    tut = np.ones(len(P), bool)
    if mm <= 0 or len(P) < 2:
        return tut
    for i in range(len(P)):
        if not tut[i]:
            continue
        d = np.linalg.norm(P[i + 1:] - P[i], axis=1)
        tut[i + 1:][(d < mm) & tut[i + 1:]] = False
    return tut


def brep_adaylari(cyl, acik, dedupe_mm=DEDUPE_MM, meta=False):
    """Silindir agizlari (iki uc, yon = +/- eksen) + aciklik merkezleri (yon = normal).

    Doner: (P, D) -- `meta=True` ise (P, D, kaynak_kaydi) ucluSU. `kaynak_kaydi`
    her aday icin onu ureten silindir/aciklik sozlugudur; agiz tanimlayicilari
    (yaricap, delik derinligi, es-eksenli kardesler) ORADAN okunur. Ayni dedupe
    maskesi uygulandigi icin siralamalar BIREBIR ortusur.

    Yon TEZIN `v_o`'suyla ayni sozlesmede: disari bakan eksen.
    """
    P, D, MET = [], [], []
    for c in cyl or []:
        ax = np.asarray(c["axis"], float)
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        ax = ax / n
        for u, s in ((c.get("mouth_a"), 1.0), (c.get("mouth_b"), -1.0)):
            if u is not None:
                P.append(np.asarray(u, float)); D.append(s * ax)
                MET.append(c)
    for o in acik or []:
        if isinstance(o, dict) and o.get("center") is not None:
            nn = np.asarray(o.get("normal", [0.0, 0.0, 1.0]), float)
            m = np.linalg.norm(nn)
            P.append(np.asarray(o["center"], float))
            D.append(nn / m if m > 1e-9 else np.array([0.0, 0.0, 1.0]))
            MET.append(o)
    if not P:
        bos = (np.zeros((0, 3)), np.zeros((0, 3)))
        return bos + ([],) if meta else bos
    P = np.asarray(P, float); D = np.asarray(D, float)
    k = _dedupe(P, dedupe_mm)
    if meta:
        return P[k], D[k], [m for m, t in zip(MET, k) if t]
    return P[k], D[k]


def birlesik_havuz(P_seg, D_seg, cyl, acik, dedupe_mm=DEDUPE_MM):
    """Segmentasyon havuzu + B-rep onerileri. Doner: (P, D, kaynak).

    `kaynak`: 0 = segmentasyon (tezin `v_o`'su), 1 = B-rep onerisi.
    Segmentasyon adaylari HER ZAMAN once gelir ve ASLA elenmez -- tezin cevabi
    havuzda bozulmadan durur.
    """
    P_seg = np.asarray(P_seg, float).reshape(-1, 3)
    D_seg = np.asarray(D_seg, float).reshape(-1, 3)
    Pb, Db = brep_adaylari(cyl, acik, dedupe_mm)
    if len(Pb) and len(P_seg):
        # segmentasyon adayina cok yakin B-rep onerisi GEREKSIZ: ayni agiz
        uz = np.linalg.norm(Pb[:, None] - P_seg[None], axis=-1).min(1)
        k = uz >= dedupe_mm
        Pb, Db = Pb[k], Db[k]
    P = np.vstack([P_seg, Pb]) if len(Pb) else P_seg
    D = np.vstack([D_seg, Db]) if len(Db) else D_seg
    kaynak = np.concatenate([np.zeros(len(P_seg), int), np.ones(len(Pb), int)])
    return P, D, kaynak


# --- MESH TEPESI KAYNAGI ---------------------------------------------------
# Olculdu (`results/tavan_080_eksensiz.json`, D7 marka-disi, isaretli aci):
#   yalniz B-rep havuzu          tavan 0.7472   102 aday/parca
#   + mesh p>=0.50, 2mm seyrelt  tavan 0.8347   318 aday/parca
#   + mesh p>=0.05, 2mm seyrelt  tavan 0.9235   829 aday/parca
# Yon kaynagi olarak tepenin YEREL NORMALI kullanilir; bu, "yon hicbir kaynakta
# yok" kovasini %23.0'ten %3.1'e indirdi.
#
# EKSEN BOYU ORNEKLEME KASTEN YOK: parca basina ~670 aday ekleyip tavani cok az
# oynatiyordu (1162 adayda 0.7798 vs 102 adayda 0.7472).
# `BH_MESH_ESIK` ile dusurulebilir. OLCULDU (D7 teshis): CWT'de konum recall
# 0.50 esiginde 0.5532, 0.05'te 0.7914. Segmentasyon zayif markalarda CP
# bolgelerindeki tepeler 0.50'yi GECEMIYOR ve havuz orayi hic gormuyor.
# Esigi dusurmek aday EKLER, asla CIKARMAZ -> havuz tavanini MONOTON yukseltir.
MESH_ESIK = float(os.environ.get("BH_MESH_ESIK", "0.50"))
MESH_DEDUPE_MM = 2.0


def tepe_normalleri(V, F):
    """Alan agirlikli tepe normalleri (agizda disari bakar)."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    N = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(n, 1e-12)


def mesh_adaylari(V, F, ppos, esik=MESH_ESIK, dedupe_mm=MESH_DEDUPE_MM):
    """p_pos esigini gecen mesh tepeleri, uzamsal seyreltmeyle.

    Doner: (P, D) -- D tepenin YEREL NORMALI (disari). Seyreltmede p_pos'u
    yuksek olan tutulur; tolerans YANAL 2mm oldugu icin 2mm'de tek tepe yeter.
    """
    V = np.asarray(V, float)
    ppos = np.asarray(ppos, float)
    k = ppos >= esik
    if not k.any():
        return np.zeros((0, 3)), np.zeros((0, 3))
    P = V[k]
    N = tepe_normalleri(V, F)[k]
    s = ppos[k]
    if dedupe_mm > 0 and len(P) > 1:
        sira = np.argsort(-s)
        tut = np.ones(len(P), bool)
        for i in sira:
            if not tut[i]:
                continue
            uz = np.linalg.norm(P - P[i], axis=1)
            yakin = (uz < dedupe_mm)
            yakin[i] = False
            tut[yakin] = False
        P, N = P[tut], N[tut]
    return P, N


def tam_havuz(P_seg, D_seg, cyl, acik, V=None, F=None, ppos=None,
              dedupe_mm=DEDUPE_MM, mesh_esik=MESH_ESIK,
              mesh_dedupe_mm=MESH_DEDUPE_MM):
    """Segmentasyon + B-rep + (varsa) mesh tepeleri. Doner: (P, D, kaynak).

    `kaynak`: 0 = segmentasyon (tezin `v_o`'su), 1 = B-rep, 2 = mesh tepesi.
    Tezin adaylari HER ZAMAN basta ve ASLA elenmez.
    """
    P, D, kay = birlesik_havuz(P_seg, D_seg, cyl, acik, dedupe_mm)
    if V is None or F is None or ppos is None:
        return P, D, kay
    Pm, Dm = mesh_adaylari(V, F, ppos, mesh_esik, mesh_dedupe_mm)
    if len(Pm) and len(P):
        uz = np.linalg.norm(Pm[:, None] - P[None], axis=-1).min(1)
        k = uz >= mesh_dedupe_mm
        Pm, Dm = Pm[k], Dm[k]
    if not len(Pm):
        return P, D, kay
    return (np.vstack([P, Pm]), np.vstack([D, Dm]),
            np.concatenate([kay, np.full(len(Pm), 2, int)]))
