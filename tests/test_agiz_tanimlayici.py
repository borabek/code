# -*- coding: utf-8 -*-
"""Agiz tanimlayici testleri: her sutunun FIZIKSEL anlamini dogrular.

Bir tanimlayici sessizce sifir/sabit donerse model onu ogrenemez ama HATA DA
VERMEZ -- bu projede daha once boyle bir kol bir geceye mal oldu. O yuzden her
sutun kendi anlamiyla ayri sinanir.
"""
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agiz_tanimlayici as AT  # noqa: E402


def kutu():
    return trimesh.creation.box((20, 20, 20))


def sil(r=2.0, a=(0, 0, 0), b=(0, 0, 10)):
    return {"radius": r, "mouth_a": list(a), "mouth_b": list(b)}


def test_sutun_adlari_ve_sayisi():
    assert len(AT.AD) == 9
    assert AT.AD[0] == "yaricap" and AT.AD[-1] == "yaricap_yuzde"


def test_bos_girdi_bos_matris():
    X = AT.tanimla(np.zeros((0, 3)), np.zeros((0, 3)), [], kutu(), 30.0)
    assert X.shape == (0, 9)


def test_yaricap_derinlik_narinlik():
    X = AT.tanimla(np.array([[0.0, 0, 0]]), np.array([[0.0, 0, -1]]),
                   [sil(2.0, (0, 0, 0), (0, 0, 10))], kutu(), 30.0)
    assert X[0, 0] == pytest.approx(2.0)
    assert X[0, 1] == pytest.approx(10.0)
    assert X[0, 2] == pytest.approx(5.0)


def test_erisim_disari_bos_ise_UZAK_doner():
    """Kutunun disinda, disari bakan agiz -> onunde hicbir sey yok."""
    X = AT.tanimla(np.array([[0.0, 0, 30]]), np.array([[0.0, 0, 1.0]]),
                   [sil(1.0, (0, 0, 30), (0, 0, 31))], kutu(), 50.0)
    assert X[0, 5] == pytest.approx(50.0)


def test_girme_govdeye_carpinca_SONLU():
    """Kutunun ustunde, govdeye dogru bakan agiz -> sonlu mesafede carpar."""
    X = AT.tanimla(np.array([[0.0, 0, 30]]), np.array([[0.0, 0, 1.0]]),
                   [sil(1.0, (0, 0, 30), (0, 0, 31))], kutu(), 50.0)
    assert 0.0 < X[0, 3] < 50.0


def test_kenar_isini_govdeyi_gorur():
    """Kenar isinlari da ayni govdeye carpmali (eksende bos + kenarda dolu ayrimi)."""
    X = AT.tanimla(np.array([[0.0, 0, 30]]), np.array([[0.0, 0, 1.0]]),
                   [sil(2.0, (0, 0, 30), (0, 0, 31))], kutu(), 50.0)
    assert 0.0 < X[0, 4] < 50.0


def test_es_eksen_kardes_sayar():
    P = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
    D = np.tile(np.array([[0.0, 0, 1.0]]), (3, 1))
    X = AT.tanimla(P, D, [sil()] * 3, kutu(), 50.0)
    assert (X[:, 6] == 2).all()


def test_yaricapi_farkli_kardes_SAYILMAZ():
    P = np.array([[0.0, 0, 0], [5.0, 0, 0]])
    D = np.tile(np.array([[0.0, 0, 1.0]]), (2, 1))
    X = AT.tanimla(P, D, [sil(1.5), sil(9.0)], kutu(), 50.0)
    assert (X[:, 6] == 0).all()


def test_aralik_duzeni_esit_araliklarda_YUKSEK():
    P = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0], [15.0, 0, 0]])
    D = np.tile(np.array([[0.0, 0, 1.0]]), (4, 1))
    X = AT.tanimla(P, D, [sil()] * 4, kutu(), 50.0)
    assert X[:, 7].max() > 0.9


def test_aralik_duzeni_duzensizde_DUSUK():
    P = np.array([[0.0, 0, 0], [1.0, 0, 0], [30.0, 0, 0], [31.0, 0, 0]])
    D = np.tile(np.array([[0.0, 0, 1.0]]), (4, 1))
    X = AT.tanimla(P, D, [sil()] * 4, kutu(), 50.0)
    assert X[:, 7].max() < 0.9


def test_yaricap_yuzdeligi_siralamayi_izler():
    P = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
    D = np.tile(np.array([[0.0, 0, 1.0]]), (3, 1))
    X = AT.tanimla(P, D, [sil(1.0), sil(2.0), sil(3.0)], kutu(), 50.0)
    assert 0.0 <= X[:, 8].min() and X[:, 8].max() <= 1.0
    assert X[2, 8] > X[0, 8]


def test_sifir_yaricap_ve_bozuk_eksen_PATLAMAZ():
    X = AT.tanimla(np.array([[0.0, 0, 30], [1.0, 0, 30]]),
                   np.array([[0.0, 0, 1.0], [0.0, 0, 0.0]]),
                   [sil(0.0), sil(1.0)], kutu(), 50.0)
    assert np.isfinite(X).all()


def test_aciklik_kaydinda_esd_r_kullanilir():
    """Duzlemsel acikliklarda `radius` yok, `esd_r` var."""
    X = AT.tanimla(np.array([[0.0, 0, 30]]), np.array([[0.0, 0, 1.0]]),
                   [{"esd_r": 1.7, "center": [0, 0, 30], "normal": [0, 0, 1]}],
                   kutu(), 50.0)
    assert X[0, 0] == pytest.approx(1.7)
