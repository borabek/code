# -*- coding: utf-8 -*-
"""KARAR OLCUTU testleri.

Bu kural, dagitim kararlarini veren tek cubuktur. En onemli testi geriye donuk olan:
GECENIN GERCEK KOLLARINI yeniden ureteibilmeli. Uretemiyorsa ya kural ya da o gece verilen
karar yanlisti -- ikisi de sessizce gecmemeli.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import karar_olcutu as K


def test_gecenin_kararlarini_YENIDEN_URETIYOR():
    assert K._kendini_sina(), "kural, 2026-08-01 kararlariyla celisiyor"


def test_isaret_yetmez_BUYUKLUK_sart():
    """t16 dersi: '+0.0018 de bir artistir ama gurultudur.'"""
    taban = {"tanidik": 0.74, "A": 0.50, "B": 0.70}
    assert not K.degerlendir(taban, {"tanidik": 0.74, "A": 0.5018, "B": 0.7018})
    # bu test BUYUKLUK sartini sinar, KANIT sartini degil -> kanit acikca kapatilir
    assert K.degerlendir(taban, {"tanidik": 0.74, "A": 0.52, "B": 0.72}, kanit_gerekli=False)


def test_riski_TASIMAK_gecmez():
    """ExtraTrees dersi: bir bolmede +0.05, digerinde -0.04 = ortalama sifir."""
    taban = {"tanidik": 0.74, "A": 0.50, "B": 0.70}
    assert not K.degerlendir(taban, {"tanidik": 0.74, "A": 0.55, "B": 0.66})


def test_ortalama_tek_bolmedeki_COKUSU_gizleyemez():
    """(4) maddesi: ortalama guzel olsa bile bir bolme cokerse gecmez."""
    taban = {"tanidik": 0.74, "A": 0.50, "B": 0.70}
    k = K.degerlendir(taban, {"tanidik": 0.74, "A": 0.70, "B": 0.60})
    assert not k and "buyuk kayip" in k.gerekce


def test_tanidik_veride_buyuk_kayip_gecmez():
    taban = {"tanidik": 0.74, "A": 0.50, "B": 0.70}
    assert not K.degerlendir(taban, {"tanidik": 0.70, "A": 0.56, "B": 0.73})


def test_ga_sifiri_iceriyorsa_GURULTU_diye_isaretlenir():
    k = K.degerlendir({"tanidik": 0.74, "A": 0.50, "B": 0.70},
                      {"tanidik": 0.74, "A": 0.53, "B": 0.72},
                      ga={"A": (-0.01, 0.07), "B": (0.005, 0.035)})
    assert "GURULTU" in k.ayrinti["    GA A"] and "GERCEK" in k.ayrinti["    GA B"]


def test_gorulmemis_bolme_yoksa_PATLAR():
    """Yalniz tanidik veriyle dagitim karari verilemez -- sessizce 'GECTI' dememeli."""
    with pytest.raises(AssertionError):
        K.degerlendir({"tanidik": 0.74}, {"tanidik": 0.75})


def test_GA_sifiri_iceriyorsa_DAGITIMA_izin_vermez():
    """2026-08-01 DENETIM BULGUSU: GA hesaplaniyor, YAZDIRILIYOR ama karara KATILMIYORDU.
    Yani gurultuden ayirt edilemeyen bir kazanc 'GECTI' alabiliyordu."""
    taban = {"tanidik": 0.74, "A": 0.50, "B": 0.70}
    aday = {"tanidik": 0.74, "A": 0.56, "B": 0.73}
    assert not K.degerlendir(taban, aday, ga={"A": (-0.01, 0.13), "B": (-0.02, 0.08)})
    assert K.degerlendir(taban, aday, ga={"A": (0.02, 0.10), "B": (0.005, 0.055)})


def test_GA_verilmezse_karar_KANITSIZ_sayilir():
    """Dagitim karari GA olmadan verilemez; tarihsel kollari yeniden uretmek icin
    kanit_gerekli=False ACIKCA istenmeli."""
    taban = {"tanidik": 0.74, "A": 0.50, "B": 0.70}
    aday = {"tanidik": 0.74, "A": 0.56, "B": 0.73}
    k = K.degerlendir(taban, aday)
    assert not k and "kanit" in k.gerekce
    assert K.degerlendir(taban, aday, kanit_gerekli=False)


def test_kanitlanmis_BUYUK_kayip_engeller():
    taban = {"tanidik": 0.74, "A": 0.50, "B": 0.70}
    aday = {"tanidik": 0.74, "A": 0.62, "B": 0.63}
    assert not K.degerlendir(taban, aday, ga={"A": (0.05, 0.19), "B": (-0.12, -0.02)})
