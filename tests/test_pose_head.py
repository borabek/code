# -*- coding: utf-8 -*-
"""POSE HEAD testleri -- gate KARARINDAN SONRA calisan lateral correction.

Tavan olcumu (results/t_tavan.json): kahin gate robot-haziri only +0.059 tasiyor, KONUM
+0.325. Yani gate'in otesindeki single real kaldirac buydu. 2026-08-01'de RULE tabanli four
konum/direction kolu was tried and dordu de became; difference, this kafanin duzeltmenin BUYUKLUGUNU candidate basina
OGRENMESI.
"""
import os 
import sys 

import numpy as np 
import pytest 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
KOK =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))


def _wg ():
    os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
    import importlib 
    import wire_gate 
    importlib .reload (wire_gate )
    return wire_gate 


def test_yerel_cerceve_ortonormal_ve_eksene_hizali ():
    wg =_wg ()
    # TOL 1e-8: 1e-9 KAYAN NOKTA gurultusune takiliyor (measured: 1.0000000827e-09).
    # Testin kendisi extra sikiydi, kod not.
    T =1e-8 
    for d in ([0 ,0 ,1.0 ],[1.0 ,0 ,0 ],[0.3 ,-0.5 ,0.81 ]):
        e ,u ,v =wg ._yerel_cerceve (np .array (d ,float ))
        assert abs (np .linalg .norm (e )-1 )<T 
        assert abs (float (e @u ))<T and abs (float (e @v ))<T 
        assert abs (float (u @v ))<T 
        assert abs (np .linalg .norm (u )-1 )<T and abs (np .linalg .norm (v )-1 )<T 


def test_duzeltme_EKSENE_DIK_kalir ():
    """Eksenel depth DEGISMEMELI -- only eksene dik deviation duzeltilir."""
    wg =_wg ()
    if wg ._load (wg .POSE_PATH )is None :
        pytest .skip ("pose modeli yok")
    d =np .array ([0.0 ,0.0 ,1.0 ])
    cps =[{"point":np .array ([1.0 ,2.0 ,5.0 ]),"direction":d }]
    onc =cps [0 ]["point"].copy ()
    out =wg .pose_correct (np .zeros ((1 ,wg ._load (wg .POSE_PATH )["n_feat"])),cps )
    kayma =np .asarray (out [0 ]["point"],float )-onc 
    assert abs (float (kayma @d ))<1e-9 ,"duzeltme EKSEN boyunca kaymis (depth degismis)"


def test_duzeltme_MAKS_MM_ile_sinirli ():
    wg =_wg ()
    m =wg ._load (wg .POSE_PATH )
    if m is None :
        pytest .skip ("pose modeli yok")
    X =np .zeros ((5 ,m ["n_feat"]))
    cps =[{"point":np .zeros (3 ),"direction":np .array ([0.0 ,0.0 ,1.0 ])}for _ in range (5 )]
    onc =[c ["point"].copy ()for c in cps ]
    out =wg .pose_correct (X ,cps )
    for c ,o in zip (out ,onc ):
        assert np .linalg .norm (np .asarray (c ["point"],float )-o )<=m ["maks_mm"]+1e-6 


def test_model_yoksa_CP_degismez ():
    wg =_wg ()
    cps =[{"point":np .array ([1.0 ,2.0 ,3.0 ]),"direction":np .array ([0.0 ,0.0 ,1.0 ])}]
    out =wg .pose_correct (np .zeros ((1 ,58 )),cps ,model_path ="results/_yok_boyle_bir_dosya.pkl")
    assert np .allclose (out [0 ]["point"],[1.0 ,2.0 ,3.0 ])


def test_yanlis_genislik_SESSIZ_gecmez_ve_CP_bozmaz ():
    """Ozellik genisligi uymuyorsa correction YAPILMAZ and sayac artar."""
    wg =_wg ()
    m =wg ._load (wg .POSE_PATH )
    if m is None :
        pytest .skip ("pose modeli yok")
    wg .fallback_ozet ()# sayaci sifirla
    cps =[{"point":np .array ([1.0 ,2.0 ,3.0 ]),"direction":np .array ([0.0 ,0.0 ,1.0 ])}]
    out =wg .pose_correct (np .zeros ((1 ,m ["n_feat"]-3 )),cps )
    assert np .allclose (out [0 ]["point"],[1.0 ,2.0 ,3.0 ])
    assert any ("pose:genislik"in k for k in wg .fallback_ozet ())


def test_urun_pose_head_i_GATE_SONRASI_cagiriyor ():
    with open (os .path .join (KOK ,"robot_cp.py"),encoding ="utf-8")as f :
        s =f .read ()
    i_gate =s .index ("robot_wire_gate")
    i_pose =s .index ("pose_correct")
    assert i_pose >i_gate ,"pose duzeltmesi gate KARARINDAN ONCE cagriliyor"
    assert '_c.get("_gate_hata")'in s or "_gate_hata"in s [:i_pose ],"gate hatasi olan parcada pose uygulanmamali"
