# -*- coding: utf-8 -*-
"""Genisletilmis urun yolu testleri.

En kritik degismezler: (1) tezin `v_o` adaylari havuzda and ONDE, (2) sign
duzeltmesi FIZIKSEL kurala uyar, (3) arm config/cevre with KAPATILABILIR,
(4) mesh tepeleri havuza GIRMEZ (three olcumde zarar verdigi for).
"""
import os 
import sys 

import numpy as np 
import pytest 
import trimesh 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import mouth_descriptor as AT # noqa: E402
import product_genis as UG # noqa: E402


def sil (c =(0 ,0 ,0 ),a =(-9 ,0 ,0 ),b =(9 ,0 ,0 ),axis =(1 ,0 ,0 ),r =1.5 ):
    return {"axis":list (axis ),"center":list (c ),"radius":r ,
    "mouth_a":list (a ),"mouth_b":list (b )}


def test_tezin_adaylari_havuzda_ve_ONDE ():
    Ps =np .array ([[0.0 ,0 ,0 ],[1.0 ,0 ,0 ]])
    Ds =np .array ([[0.0 ,0 ,1.0 ],[0.0 ,0 ,1.0 ]])
    P ,D ,kay =UG .pool (Ps ,Ds ,[sil ((50 ,0 ,0 ),(45 ,0 ,0 ),(55 ,0 ,0 ))],[])
    assert np .allclose (P [:2 ],Ps )
    assert list (kay [:2 ])==[0 ,0 ]


def test_mesh_tepeleri_havuza_GIRMEZ ():
    """Uc bagimsiz olcumde zarar verdi; urun yolunda BULUNMAMALI."""
    Ps =np .array ([[0.0 ,0 ,0 ]])
    Ds =np .array ([[0.0 ,0 ,1.0 ]])
    _P ,_D ,kay =UG .pool (Ps ,Ds ,[sil ()],[])
    assert set (np .unique (kay ).tolist ())<={0 ,1 }


def test_isaret_duzeltme_fiziksel_kural ():
    D =np .array ([[0.0 ,0 ,1.0 ],[0.0 ,0 ,1.0 ]])
    T =np .zeros ((2 ,len (AT .AD )))
    T [0 ,UG .GIRME ]=10.0 # iceri open, disari closed -> TERS
    T [0 ,UG .ERISIM ]=1.0 
    T [1 ,UG .GIRME ]=1.0 # correct direction -> DEGISMEZ
    T [1 ,UG .ERISIM ]=10.0 
    Y =UG .isaret_duzelt (D ,T )
    assert np .allclose (Y [0 ],[0 ,0 ,-1 ])
    assert np .allclose (Y [1 ],[0 ,0 ,1 ])


def test_isaret_duzeltme_bos_girdide_PATLAMAZ ():
    assert len (UG .isaret_duzelt (np .zeros ((0 ,3 )),np .zeros ((0 ,9 ))))==0 


def test_kol_cevre_degiskeniyle_kapanir ():
    old_ =os .environ .get ("URUN_GENIS")
    try :
        os .environ ["URUN_GENIS"]="0"
        assert UG ._cfg ("robot_genis_havuz","URUN_GENIS",True )is False 
        os .environ ["URUN_GENIS"]="1"
        assert UG ._cfg ("robot_genis_havuz","URUN_GENIS",True )is True 
    finally :
        if old_ is None :
            os .environ .pop ("URUN_GENIS",None )
        else :
            os .environ ["URUN_GENIS"]=old_ 


def test_tanimlayici_sutun_sayisi ():
    m =trimesh .creation .box ((20 ,20 ,20 ))
    T =UG .tanimlayici (np .array ([[0.0 ,0 ,10 ]]),np .array ([[0.0 ,0 ,1.0 ]]),
    [sil ()],m ,35.0 )
    assert T .shape ==(1 ,len (AT .AD ))


def test_tanimlayici_silindirsiz_PATLAMAZ ():
    m =trimesh .creation .box ((20 ,20 ,20 ))
    T =UG .tanimlayici (np .array ([[0.0 ,0 ,10 ]]),np .array ([[0.0 ,0 ,1.0 ]]),
    None ,m ,35.0 )
    assert T .shape ==(1 ,len (AT .AD ))and np .isfinite (T ).all ()


def test_sec_tek_adayda_DEGISTIRMEZ ():
    m =trimesh .creation .box ((20 ,20 ,20 ))
    P =np .array ([[0.0 ,0 ,10.0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ]])
    P2 ,D2 =UG .sec (P ,D ,np .zeros ((1 ,58 )),[sil ()],m ,35.0 ,None )
    assert np .allclose (P2 ,P )and np .allclose (D2 ,D )


def test_esik_D6da_secildi_ve_SABIT ():
    """0.05 `results/secici_ailesi.json` inside D6'da secildi, D7'de taranmadi."""
    assert UG .ESIK ==pytest .approx (0.05 )


def test_girme_erisim_indisleri_AD_ile_tutarli ():
    assert AT .AD [UG .GIRME ]=="girme"
    assert AT .AD [UG .ERISIM ]=="erisim"
