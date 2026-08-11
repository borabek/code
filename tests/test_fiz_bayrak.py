# -*- coding: utf-8 -*-
"""FIZIKSEL KUSUR BAYRAKLARI -- SENTETIK KALIBRASYON (2026-08-05).

NEDEN VAR: ilk surum sessizce YANLIS renkler ciziyordu ve gercek parcada "0 kusur"
dedigi icin FARK EDILMIYORDU. Kalibrasyon iki hatayi yakaladi:
  1. `ters_yon` "p+2mm*d iceride mi" diye soruyordu -> nokta zaten govde icindeyse HER
     ZAMAN dogru cikip `govde_ici`nin kopyasini uretiyor, disaridaki GERCEK ters yonu
     ise kaciriyordu (2mm yuzeye ulasmiyor).
  2. `onu_kapali` `exit_length` kullaniyordu ve yuzeye 0.5mm'de bile ATESLEMIYORDU.
Ikisi de dogrudan isin mesafesiyle yeniden yazildi.

CEVABI BILINEN geometri: 20mm kup (yuzeyler +-10mm). Her vakanin dogru cevabi
elle hesaplanabilir -- "test gecsin diye" ayarlanmis bir beklenti YOK.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def kup():
    import trimesh
    return trimesh.creation.box(extents=(20.0, 20.0, 20.0))


def _b(kup, p, d):
    from export_robot_glb import _fiz_bayraklar
    return {a for a, _ in _fiz_bayraklar(kup, np.array(p, float), np.array(d, float))}


def test_govde_ici_atesler(kup):
    """Kup merkezi govdenin icinde -- ve YALNIZ govde_ici olmali (ters_yon KOPYASI DEGIL)."""
    assert _b(kup, [0, 0, 0], [0, 0, 1]) == {"govde_ici"}


def test_temiz_nokta_bayraksiz(kup):
    """Disarida, disari bakan, uzak -- hicbir kusur yok."""
    assert _b(kup, [0, 0, 15], [0, 0, 1]) == set()


def test_ters_yon_disaridan_yakalanir(kup):
    """Disaridaki nokta govdeye BAKIYORSA ters_yon atesler (ilk surum bunu KACIRIYORDU)."""
    assert "ters_yon" in _b(kup, [0, 0, 30], [0, 0, -1])


def test_onu_kapali_yuzeye_yakinken(kup):
    """Yuzeye 0.5mm: hem ters_yon hem onu_kapali."""
    assert _b(kup, [0, 0, 10.5], [0, 0, -1]) == {"ters_yon", "onu_kapali"}


def test_esik_siniri_bes_mm_gecer(kup):
    """TAM 5mm bosluk VARDIR -> onu_kapali ATESLEMEZ (kural '< 5.0').

    Bu vaka ilk kalibrasyonda 'kaldi' gorunuyordu ama olculdu: isin tam 5.000mm.
    Kod dogru, BEKLENTI yanlisti. Sinir davranisi burada KILITLENIYOR."""
    g = _b(kup, [0, 0, 15], [0, 0, -1])
    assert "ters_yon" in g and "onu_kapali" not in g


def test_dar_agiz_delikte_atesler():
    """Telden (1.78mm) dar bir delikte dar_agiz atesler; genis delikte atesLEMEZ."""
    import trimesh
    from export_robot_glb import _fiz_bayraklar
    for cap, bekle in ((1.0, True), (6.0, False)):
        kutu = trimesh.creation.box(extents=(20.0, 20.0, 10.0))
        sil = trimesh.creation.cylinder(radius=cap / 2.0, height=30.0, sections=32)
        try:
            m = kutu.difference(sil)
        except Exception:
            pytest.skip("boolean motoru yok")
        if m is None or not len(getattr(m, "faces", [])):
            pytest.skip("boolean basarisiz")
        # z=5.0 AGIZ DUZLEMI (kutu 10mm yuksek -> z in [-5,+5]). Ilk surumde z=8 verilmisti,
        # yani nokta govdenin TAMAMEN disinda kaliyordu ve olculecek agiz yoktu.
        g = {a for a, _ in _fiz_bayraklar(m, np.array([0.0, 0.0, 5.0]),
                                          np.array([0.0, 0.0, -1.0]))}
        assert ("dar_agiz" in g) == bekle, f"cap {cap}mm -> {g}"
