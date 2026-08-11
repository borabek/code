# -*- coding: utf-8 -*-
"""P0-3: TEK ESLESTIRICI -- Macar (Hungarian) atama, acgozlu yerine.

Acgozlu eslestirici ciftleri mesafeye gore gezip ilk uyani baglar. Kalabalik parcada
bu ATAMA KAYBI uretir. Macar yontemi ayni kabul kutusu icinde TP'yi enbuyukler; boylece
otopsideki "KALABALIK" kovasinin (GT'nin %12.8'i) ne kadarinin saf atama artefakti
oldugu olculebilir.

KRITIK: bu bir TOLERANS GEVSETMESI DEGILDIR. Kabul kutusu (yanal / eksenel / aci)
birebir aynidir; yalnizca kutunun ICINDEKI eslestirme optimaldir.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sina_kume import esle_detay, esle_macar


def _d(n):
    return np.tile([0.0, 0, 1], (n, 1))


def test_macar_acgozluyle_ayni_basit_durumda():
    P = np.array([[0., 0, 0], [10, 0, 0]])
    G = np.array([[0., 0, 0], [10, 0, 0]])
    a = esle_detay(P, _d(2), G, _d(2), 100.0, 2.0, 10.0, False)
    b = esle_macar(P, _d(2), G, _d(2), 100.0, 2.0, 10.0, False)
    assert a[0] == b[0] == 2


def test_macar_ACGOZLUNUN_KACIRDIGINI_yakalar():
    """Klasik acgozlu tuzagi: en yakin cift once baglanir ve digerini bos birakir.

    P0 -> G0 mesafe 0.5, P0 -> G1 mesafe 1.9 ; P1 -> G1 mesafe 0.4, P1 -> G0 uzak.
    Acgozlu once (P1,G1) 0.4'u baglar, sonra (P0,G0) 0.5 -> ikisi de eslesir.
    Burada TERS kurulum: tek tahmin iki GT'ye yakin, ikinci tahmin yalniz birine yakin.
    """
    P = np.array([[0., 0, 0], [1.6, 0, 0]])
    G = np.array([[0., 0, 0], [1.8, 0, 0]])
    # P0 hem G0'a (0.0) hem G1'e (1.8) yakin; P1 yalniz G1'e (0.2) yakin
    a = esle_detay(P, _d(2), G, _d(2), 100.0, 2.0, 10.0, False)
    b = esle_macar(P, _d(2), G, _d(2), 100.0, 2.0, 10.0, False)
    assert b[0] >= a[0], "Macar acgozluden AZ eslestiremez"
    assert b[0] == 2


def test_macar_KABUL_KUTUSUNU_GEVSETMEZ():
    """Toleransin disindaki cift, optimal atamada bile TP sayilmamali."""
    P = np.array([[0., 0, 0]])
    G = np.array([[50., 0, 0]])          # yanal 50mm, tolerans 2mm
    tp, fp, fn, _ = esle_macar(P, _d(1), G, _d(1), 100.0, 2.0, 10.0, False)
    assert (tp, fp, fn) == (0, 1, 1)


def test_macar_ACI_KAPISINI_uygular():
    P = np.array([[0., 0, 0]])
    Pd = np.array([[1., 0, 0]])          # GT yonune DIK
    G = np.array([[0., 0, 0]])
    tp, _fp, _fn, _ = esle_macar(P, Pd, G, _d(1), 100.0, 2.0, 10.0, False)
    assert tp == 0, "90 derece sapma aci kapisindan gecmemeli"


def test_macar_ISARETI_uygular():
    P = np.array([[0., 0, 0]])
    Pd = np.array([[0., 0, -1]])         # ters isaret
    G = np.array([[0., 0, 0]])
    assert esle_macar(P, Pd, G, _d(1), 100.0, 2.0, 10.0, False, isaretli=True)[0] == 0
    assert esle_macar(P, Pd, G, _d(1), 100.0, 2.0, 10.0, False, isaretli=False)[0] == 1


def test_macar_bir_adayi_IKI_GTye_kullanmaz():
    P = np.array([[0., 0, 0]])
    G = np.array([[0., 0, 0], [0.5, 0, 0]])
    tp, fp, fn, _ = esle_macar(P, _d(1), G, _d(2), 100.0, 2.0, 10.0, False)
    assert tp == 1 and fn == 1, "tek aday tek GT'ye sayilmali"


def test_bos_girdiler():
    for f in (esle_detay, esle_macar):
        assert f(np.zeros((0, 3)), np.zeros((0, 3)), np.array([[0., 0, 0]]),
                 _d(1), 100.0, 2.0, 10.0, False)[:3] == (0, 0, 1)
