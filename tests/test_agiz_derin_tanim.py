# -*- coding: utf-8 -*-
"""Ikinci kademe mouth olculeri: each sutunun FIZIKSEL anlamini dogrular.

Sessizce sabit/sifir donen a column modele bilgi tasimaz but HATA DA VERMEZ;
that yuzden each olcu own anlamiyla ayri sinanir.
"""
import os 
import sys 

import numpy as np 
import pytest 
import trimesh 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import agiz_derin_tanim as AD # noqa: E402


def kutu (k =20.0 ):
    return trimesh .creation .box ((k ,k ,k ))


def delikli_kutu (r =2.0 ,k =20.0 ):
    """Z ekseni along bastan sona R yaricapli hole acilmis kutu."""
    b =trimesh .creation .box ((k ,k ,k ))
    c =trimesh .creation .cylinder (radius =r ,height =k *2.0 ,sections =48 )
    return trimesh .boolean .difference ([b ,c ])


def test_sutun_sayisi ():
    assert len (AD .AD )==9 


def test_bos_girdi ():
    X =AD .tanimla (np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),kutu (),30.0 )
    assert X .shape ==(0 ,9 )


def test_bozuk_yon_PATLAMAZ ():
    X =AD .tanimla (np .array ([[0.0 ,0 ,10 ]]),np .array ([[0.0 ,0 ,0 ]]),
    kutu (),30.0 )
    assert np .isfinite (X ).all ()


def test_delikte_profil_ACIK_kalir ():
    """Gercek kanalda serbest radius depth along ~sabit and r'ye yakin."""
    m =delikli_kutu (r =2.0 )
    X =AD .tanimla (np .array ([[0.0 ,0 ,10.0 ]]),np .array ([[0.0 ,0 ,1.0 ]]),
    m ,35.0 )
    assert 1.0 <X [0 ,0 ]<4.0 # profil_ort hole yaricapi mertebesinde
    assert X [0 ,2 ]>0.7 # daralma absent


def test_duz_yuzeyde_profil_KAPALI ():
    """Deliksiz kutunun yuzunde iceri bakan mouth -> serbest radius small."""
    m =kutu ()
    X =AD .tanimla (np .array ([[0.0 ,0 ,10.0 ]]),np .array ([[0.0 ,0 ,1.0 ]]),
    m ,35.0 )
    d =AD .tanimla (np .array ([[0.0 ,0 ,10.0 ]]),np .array ([[0.0 ,0 ,1.0 ]]),
    delikli_kutu (r =2.0 ),35.0 )
    assert X [0 ,0 ]<d [0 ,0 ]# delikli which is DAHA OPEN


def test_karsi_agiz_delikte_1_kutuda_0 ():
    dl =AD .tanimla (np .array ([[0.0 ,0 ,10.0 ]]),np .array ([[0.0 ,0 ,1.0 ]]),
    delikli_kutu (r =2.0 ),35.0 )
    kt =AD .tanimla (np .array ([[0.0 ,0 ,10.0 ]]),np .array ([[0.0 ,0 ,1.0 ]]),
    kutu (),35.0 )
    assert dl [0 ,6 ]==pytest .approx (1.0 )
    assert kt [0 ,6 ]==pytest .approx (0.0 )


def test_govde_orani_merkezde_kucuk_kenarda_buyuk ():
    m =kutu ()
    a =AD .tanimla (np .array ([[0.0 ,0 ,0.0 ]]),np .array ([[0.0 ,0 ,1.0 ]]),m ,35.0 )
    b =AD .tanimla (np .array ([[0.0 ,0 ,10.0 ]]),np .array ([[0.0 ,0 ,1.0 ]]),m ,35.0 )
    assert a [0 ,8 ]<b [0 ,8 ]


def test_ciktilar_sonlu_ve_negatif_degil ():
    m =delikli_kutu (r =1.5 )
    P =np .array ([[0.0 ,0 ,10.0 ],[0.0 ,0 ,-10.0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ],[0.0 ,0 ,-1.0 ]])
    X =AD .tanimla (P ,D ,m ,35.0 )
    assert np .isfinite (X ).all ()
    for j in (0 ,1 ,2 ,3 ,4 ,7 ,8 ):
        assert (X [:,j ]>=0 ).all ()


def test_halka_duzlugu_duz_yuzde_DUSUK ():
    X =AD .tanimla (np .array ([[0.0 ,0 ,10.0 ]]),np .array ([[0.0 ,0 ,1.0 ]]),
    kutu (),35.0 )
    assert X [0 ,7 ]<0.5 
