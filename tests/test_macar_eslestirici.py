# -*- coding: utf-8 -*-
"""P0-3: TEK ESLESTIRICI -- Macar (Hungarian) atama, acgozlu instead of.

Acgozlu eslestirici ciftleri mesafeye according to gezip first uyani baglar. Kalabalik parcada
this ATAMA KAYBI produces. Macar yontemi same kabul kutusu inside TP'yi enbuyukler; so
otopsideki "KALABALIK" kovasinin (GT'nin %12.8'i) ne kadarinin saf atama artefakti
oldugu olculebilir.

KRITIK: this a TOLERANS GEVSETMESI DEGILDIR. Kabul kutusu (lateral / axial / angle)
birebir aynidir; only kutunun ICINDEKI eslestirme optimaldir.
"""
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
from sina_cluster import match_greedy ,match_hungarian 


def _d (n ):
    return np .tile ([0.0 ,0 ,1 ],(n ,1 ))


def test_macar_acgozluyle_ayni_basit_durumda ():
    P =np .array ([[0. ,0 ,0 ],[10 ,0 ,0 ]])
    G =np .array ([[0. ,0 ,0 ],[10 ,0 ,0 ]])
    a =match_greedy (P ,_d (2 ),G ,_d (2 ),100.0 ,2.0 ,10.0 ,False )
    b =match_hungarian (P ,_d (2 ),G ,_d (2 ),100.0 ,2.0 ,10.0 ,False )
    assert a [0 ]==b [0 ]==2 


def test_macar_ACGOZLUNUN_KACIRDIGINI_yakalar ():
    """Klasik acgozlu tuzagi: most yakin double before baglanir and digerini empty birakir.

    P0 -> G0 distance 0.5, P0 -> G1 distance 1.9 ; P1 -> G1 distance 0.4, P1 -> G0 uzak.
    Acgozlu before (P1,G1) 0.4'u baglar, after (P0,G0) 0.5 -> ikisi de eslesir.
    Burada TERS kurulum: single prediction two GT'ye yakin, ikinci prediction only birine yakin.
    """
    P =np .array ([[0. ,0 ,0 ],[1.6 ,0 ,0 ]])
    G =np .array ([[0. ,0 ,0 ],[1.8 ,0 ,0 ]])
    # P0 hem G0'a (0.0) hem G1'e (1.8) yakin; P1 only G1'e (0.2) yakin
    a =match_greedy (P ,_d (2 ),G ,_d (2 ),100.0 ,2.0 ,10.0 ,False )
    b =match_hungarian (P ,_d (2 ),G ,_d (2 ),100.0 ,2.0 ,10.0 ,False )
    assert b [0 ]>=a [0 ],"Macar acgozluden AZ eslestiremez"
    assert b [0 ]==2 


def test_macar_KABUL_KUTUSUNU_GEVSETMEZ ():
    """Toleransin disindaki double, optimal atamada bile TP sayilmamali."""
    P =np .array ([[0. ,0 ,0 ]])
    G =np .array ([[50. ,0 ,0 ]])# lateral 50mm, tolerans 2mm
    tp ,fp ,fn ,_ =match_hungarian (P ,_d (1 ),G ,_d (1 ),100.0 ,2.0 ,10.0 ,False )
    assert (tp ,fp ,fn )==(0 ,1 ,1 )


def test_macar_ACI_KAPISINI_uygular ():
    P =np .array ([[0. ,0 ,0 ]])
    Pd =np .array ([[1. ,0 ,0 ]])# GT yonune DIK
    G =np .array ([[0. ,0 ,0 ]])
    tp ,_fp ,_fn ,_ =match_hungarian (P ,Pd ,G ,_d (1 ),100.0 ,2.0 ,10.0 ,False )
    assert tp ==0 ,"90 derece deviation aci kapisindan gecmemeli"


def test_macar_ISARETI_uygular ():
    P =np .array ([[0. ,0 ,0 ]])
    Pd =np .array ([[0. ,0 ,-1 ]])# ters sign
    G =np .array ([[0. ,0 ,0 ]])
    assert match_hungarian (P ,Pd ,G ,_d (1 ),100.0 ,2.0 ,10.0 ,False ,signed =True )[0 ]==0 
    assert match_hungarian (P ,Pd ,G ,_d (1 ),100.0 ,2.0 ,10.0 ,False ,signed =False )[0 ]==1 


def test_macar_bir_adayi_IKI_GTye_kullanmaz ():
    P =np .array ([[0. ,0 ,0 ]])
    G =np .array ([[0. ,0 ,0 ],[0.5 ,0 ,0 ]])
    tp ,fp ,fn ,_ =match_hungarian (P ,_d (1 ),G ,_d (2 ),100.0 ,2.0 ,10.0 ,False )
    assert tp ==1 and fn ==1 ,"tek candidate tek GT'ye sayilmali"


def test_bos_girdiler ():
    for f in (match_greedy ,match_hungarian ):
        assert f (np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),np .array ([[0. ,0 ,0 ]]),
        _d (1 ),100.0 ,2.0 ,10.0 ,False )[:3 ]==(0 ,0 ,1 )
