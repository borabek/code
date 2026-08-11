# -*- coding: utf-8 -*-
"""Gate sonrasi kalabalik bastirma (NMS) testleri. Bkz. results/nms_tarama.json."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wire_gate  # noqa: E402


def cp(x, s):
    return {"point": [float(x), 0.0, 0.0], "wire_score": float(s)}


def test_yakin_ciftte_yuksek_skorlu_kalir():
    out = wire_gate.kalabalik_bastir([cp(0, 0.3), cp(2, 0.9)], r_mm=6.0)
    assert len(out) == 1 and out[0]["wire_score"] == pytest.approx(0.9)


def test_uzak_cift_ikisi_de_kalir():
    out = wire_gate.kalabalik_bastir([cp(0, 0.3), cp(20, 0.9)], r_mm=6.0)
    assert len(out) == 2


def test_sinir_tam_yaricapta_BASTIRILMAZ():
    """Kural `< r`; tam r mesafesi ayri agiz sayilir."""
    assert len(wire_gate.kalabalik_bastir([cp(0, 0.3), cp(6, 0.9)], r_mm=6.0)) == 2
    assert len(wire_gate.kalabalik_bastir([cp(0, 0.3), cp(5.99, 0.9)], r_mm=6.0)) == 1


def test_olcum_ve_urun_yolu_AYNI_maskeyi_alir():
    """`kalabalik_maskesi` TEK KAYNAK; `kalabalik_bastir` onu cagirir."""
    girdi = [cp(0, 0.5), cp(4, 0.9), cp(40, 0.6)]
    m = wire_gate.kalabalik_maskesi([c["point"] for c in girdi],
                                    [c["wire_score"] for c in girdi], 6.0)
    assert [c["point"][0] for c in wire_gate.kalabalik_bastir(girdi, r_mm=6.0)] ==            [c["point"][0] for c, k in zip(girdi, m) if k]


def test_sifir_yaricap_KAPATIR():
    girdi = [cp(0, 0.3), cp(1, 0.9)]
    assert wire_gate.kalabalik_bastir(girdi, r_mm=0.0) == girdi


def test_girdi_sirasi_korunur():
    out = wire_gate.kalabalik_bastir([cp(0, 0.9), cp(30, 0.5), cp(60, 0.7)], r_mm=6.0)
    assert [c["point"][0] for c in out] == [0.0, 30.0, 60.0]


def test_zincir_ortadaki_en_yuksekten_bastirilir():
    """0-4-8: 4 en yuksek. 0 ve 8 ona 6mm'den yakin -> ikisi de duser."""
    out = wire_gate.kalabalik_bastir([cp(0, 0.5), cp(4, 0.9), cp(8, 0.6)], r_mm=6.0)
    assert [c["point"][0] for c in out] == [4.0]


def test_tek_cp_ve_bos_liste():
    assert len(wire_gate.kalabalik_bastir([cp(0, 0.5)], r_mm=6.0)) == 1
    assert wire_gate.kalabalik_bastir([], r_mm=6.0) == []


def test_varsayilan_yaricap_konfigden_5mm():
    """r=5.0 DAGITILAN deger. Kural: hicbir markayi yikmayan en buyuk yaricap.
    r=6 D7'de daha yuksek robot verir ama CEM markasini 0.0164 -> 0.0000 yikar.
    Bkz. results/urun_nms_uctan_uca*.json."""
    assert wire_gate.NMS_MM == pytest.approx(5.0)
