# -*- coding: utf-8 -*-
"""topoloji_ailesi: sentetik klemens dizisi on elle dogrulama."""
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))

import topoloji_ailesi as TA # noqa: E402


def _sahne ():
    """5'li tel girisi dizisi (+z bakar), each birinin 10mm arkasinda vida yuvasi,
    a de uzakta single basina duran SAHTE candidate."""
    on =np .array ([[5.0 *i ,0.0 ,0.0 ]for i in range (5 )])
    arka =on +np .array ([0.0 ,0.0 ,-10.0 ])
    sahte =np .array ([[100.0 ,60.0 ,40.0 ]])
    return on ,np .vstack ([on ,arka ,sahte ])


def test_dizi_uyesi_ile_yalniz_aday_ayrilir ():
    on ,Pu =_sahne ()
    z =np .tile ([0.0 ,0.0 ,1.0 ],(2 ,1 ))
    P =np .vstack ([on [0 ],[100.0 ,60.0 ,40.0 ]])
    X =TA .oznitelik (P ,z ,Pu ,diag =200.0 )
    ad ={a :i for i ,a in enumerate (TA .OZ_AD )}
    # array uyesi: own duzleminde 4 komsu, ekseninde 1 (arkadaki vida yuvasi)
    assert X [0 ,ad ["dik_dizi_n"]]==4 ,X [0 ]
    assert X [0 ,ad ["es_eksen_n"]]==1 ,X [0 ]
    # only fake candidate: ikisi de sifir
    assert X [1 ,ad ["dik_dizi_n"]]==0 ,X [1 ]
    assert X [1 ,ad ["es_eksen_n"]]==0 ,X [1 ]


def test_duzenli_aralik_cv_sifir ():
    on ,Pu =_sahne ()
    X =TA .oznitelik (on [:1 ],np .array ([[0.0 ,0.0 ,1.0 ]]),Pu ,diag =200.0 )
    # dik mesafeler 5,10,15,20 -> farklar 5,5,5 -> CV = 0
    assert X [0 ,TA .OZ_AD .index ("aralik_cv")]<1e-9 ,X [0 ]


def test_duzensiz_aralik_cv_buyur ():
    Pu =np .array ([[0. ,0. ,0. ],[1. ,0. ,0. ],[2.5 ,0. ,0. ],[20. ,0. ,0. ]])
    X =TA .oznitelik (Pu [:1 ],np .array ([[0.0 ,0.0 ,1.0 ]]),Pu ,diag =200.0 )
    assert X [0 ,TA .OZ_AD .index ("aralik_cv")]>0.3 ,X [0 ]


def test_disa_bakis_isareti ():
    on ,Pu =_sahne ()
    P =np .tile (on [2 ],(2 ,1 ))# dizinin ortasi
    D =np .array ([[0. ,0. ,1. ],[0. ,0. ,-1. ]])# disari / iceri
    X =TA .oznitelik (P ,D ,Pu ,diag =200.0 )
    i =TA .OZ_AD .index ("disa_bakis")
    assert X [0 ,i ]>0 >X [1 ,i ],X [:,i ]


def test_bos_girdi_cokmez ():
    X =TA .oznitelik (np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),1.0 )
    assert X .shape ==(0 ,len (TA .OZ_AD ))


if __name__ =="__main__":
    for ad ,f in sorted (globals ().items ()):
        if ad .startswith ("test_"):
            f ()
            print (f"  GECTI  {ad }")
    print ("hepsi gecti")
