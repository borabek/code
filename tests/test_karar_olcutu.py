# -*- coding: utf-8 -*-
"""DECISION OLCUTU testleri.

Bu rule, dagitim kararlarini veren single cubuktur. En onemli testi geriye donuk which is:
GECENIN GERCEK KOLLARINI yeniden ureteibilmeli. Uretemiyorsa ya rule ya da that gece verilen
karar yanlisti -- ikisi de sessizce gecmemeli.
"""
import os 
import sys 

import pytest 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import karar_olcutu as K 


def test_gecenin_kararlarini_YENIDEN_URETIYOR ():
    assert K ._kendini_sina (),"kural, 2026-08-01 kararlariyla celisiyor"


def test_isaret_yetmez_BUYUKLUK_sart ():
    """t16 dersi: '+0.0018 de a artistir but gurultudur.'"""
    baseline ={"tanidik":0.74 ,"A":0.50 ,"B":0.70 }
    assert not K .degerlendir (baseline ,{"tanidik":0.74 ,"A":0.5018 ,"B":0.7018 })
    # this test BUYUKLUK sartini sinar, KANIT sartini not -> evidence acikca kapatilir
    assert K .degerlendir (baseline ,{"tanidik":0.74 ,"A":0.52 ,"B":0.72 },kanit_gerekli =False )


def test_riski_TASIMAK_gecmez ():
    """ExtraTrees dersi: a bolmede +0.05, digerinde -0.04 = mean sifir."""
    baseline ={"tanidik":0.74 ,"A":0.50 ,"B":0.70 }
    assert not K .degerlendir (baseline ,{"tanidik":0.74 ,"A":0.55 ,"B":0.66 })


def test_ortalama_tek_bolmedeki_COKUSU_gizleyemez ():
    """(4) maddesi: mean guzel olsa bile a split cokerse gecmez."""
    baseline ={"tanidik":0.74 ,"A":0.50 ,"B":0.70 }
    k =K .degerlendir (baseline ,{"tanidik":0.74 ,"A":0.70 ,"B":0.60 })
    assert not k and "buyuk loss"in k .rationale 


def test_tanidik_veride_buyuk_kayip_gecmez ():
    baseline ={"tanidik":0.74 ,"A":0.50 ,"B":0.70 }
    assert not K .degerlendir (baseline ,{"tanidik":0.70 ,"A":0.56 ,"B":0.73 })


def test_ga_sifiri_iceriyorsa_GURULTU_diye_isaretlenir ():
    k =K .degerlendir ({"tanidik":0.74 ,"A":0.50 ,"B":0.70 },
    {"tanidik":0.74 ,"A":0.53 ,"B":0.72 },
    ga ={"A":(-0.01 ,0.07 ),"B":(0.005 ,0.035 )})
    assert "GURULTU"in k .ayrinti ["    GA A"]and "GERCEK"in k .ayrinti ["    GA B"]


def test_gorulmemis_bolme_yoksa_PATLAR ():
    """Yalniz tanidik veriyle dagitim karari verilemez -- sessizce 'GECTI' dememeli."""
    with pytest .raises (AssertionError ):
        K .degerlendir ({"tanidik":0.74 },{"tanidik":0.75 })


def test_GA_sifiri_iceriyorsa_DAGITIMA_izin_vermez ():
    """2026-08-01 DENETIM BULGUSU: GA is computed, YAZDIRILIYOR but karara KATILMIYORDU.
    Yani gurultuden ayirt edilemeyen a kazanc 'GECTI' alabiliyordu."""
    baseline ={"tanidik":0.74 ,"A":0.50 ,"B":0.70 }
    candidate ={"tanidik":0.74 ,"A":0.56 ,"B":0.73 }
    assert not K .degerlendir (baseline ,candidate ,ga ={"A":(-0.01 ,0.13 ),"B":(-0.02 ,0.08 )})
    assert K .degerlendir (baseline ,candidate ,ga ={"A":(0.02 ,0.10 ),"B":(0.005 ,0.055 )})


def test_GA_verilmezse_karar_KANITSIZ_sayilir ():
    """Dagitim karari GA olmadan verilemez; tarihsel kollari yeniden uretmek for
    kanit_gerekli=False ACIKCA istenmeli."""
    baseline ={"tanidik":0.74 ,"A":0.50 ,"B":0.70 }
    candidate ={"tanidik":0.74 ,"A":0.56 ,"B":0.73 }
    k =K .degerlendir (baseline ,candidate )
    assert not k and "evidence"in k .rationale 
    assert K .degerlendir (baseline ,candidate ,kanit_gerekli =False )


def test_kanitlanmis_BUYUK_kayip_engeller ():
    baseline ={"tanidik":0.74 ,"A":0.50 ,"B":0.70 }
    candidate ={"tanidik":0.74 ,"A":0.62 ,"B":0.63 }
    assert not K .degerlendir (baseline ,candidate ,ga ={"A":(0.05 ,0.19 ),"B":(-0.12 ,-0.02 )})
