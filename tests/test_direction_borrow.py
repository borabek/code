# -*- coding: utf-8 -*-
"""Yon odunc alma testleri.

En kritik degismezler: (1) KONUM never does not change, (2) tezin yonu always 0.
option and esitlikte KAZANIR, (3) candidate SAYISI does not change.
"""
import os 
import sys 

import numpy as np 
import pytest 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import direction_borrow as YO # noqa: E402


def sil (c =(0 ,0 ,0 ),axis =(1 ,0 ,0 )):
    return {"axis":list (axis ),"center":list (c ),"radius":1.5 ,
    "mouth_a":[0 ,0 ,0 ],"mouth_b":[1 ,0 ,0 ]}


def test_mevcut_yon_HER_ZAMAN_sifirinci_secenek ():
    P =np .array ([[0.0 ,0 ,0 ],[3.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ],[1.0 ,0 ,0.0 ]])
    V ,X =YO .options (P ,D ,0 ,[],0.5 ,YO .baskin_yon (D ))
    assert np .allclose (V [0 ],[0 ,0 ,1 ])
    assert X [0 ,0 ]==1.0 


def test_komsunun_yonu_secenege_girer ():
    P =np .array ([[0.0 ,0 ,0 ],[3.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ],[1.0 ,0 ,0.0 ]])
    V ,_ =YO .options (P ,D ,0 ,[],0.5 ,YO .baskin_yon (D ))
    assert any (abs (abs (float (v @np .array ([1.0 ,0 ,0 ])))-1 )<1e-6 for v in V )


def test_uzak_komsunun_yonu_ALINMAZ ():
    P =np .array ([[0.0 ,0 ,0 ],[500.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ],[1.0 ,0 ,0.0 ]])
    V ,_ =YO .options (P ,D ,0 ,[],0.5 ,YO .baskin_yon (D ),komsu_r =10.0 )
    assert not any (abs (abs (float (v @np .array ([1.0 ,0 ,0 ])))-1 )<1e-6 for v in V )


def test_silindir_ekseni_IKI_isaretle_girer ():
    P =np .array ([[0.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ]])
    V ,_ =YO .options (P ,D ,0 ,[sil ((1 ,0 ,0 ),(1 ,0 ,0 ))],0.5 ,None )
    hiz =[v for v in V if abs (abs (float (v @np .array ([1.0 ,0 ,0 ])))-1 )<1e-6 ]
    assert len (hiz )>=1 


def test_ayni_yon_TEKRAR_EDILMEZ ():
    """Birbirine 5 dereceden yakin options single sayilir."""
    P =np .array ([[0.0 ,0 ,0 ],[1.0 ,0 ,0 ],[2.0 ,0 ,0 ]])
    D =np .tile (np .array ([[0.0 ,0 ,1.0 ]]),(3 ,1 ))
    V ,_ =YO .options (P ,D ,0 ,[],0.5 ,YO .baskin_yon (D ))
    assert len (V )==1 


def test_KONUM_asla_degismez_ve_SAYI_sabit ():
    P =np .array ([[0.0 ,0 ,0 ],[3.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ],[1.0 ,0 ,0.0 ]])
    P0 =P .copy ()
    new_ =YO .uygula (P ,D ,[],[0.9 ,0.9 ],lambda X :np .arange (len (X )))
    assert np .allclose (P ,P0 )
    assert len (new_ )==len (D )


def test_esitlikte_MEVCUT_kazanir ():
    """Tezin cevabi default kalmali: new direction however KESIN more iyiyse alinir."""
    P =np .array ([[0.0 ,0 ,0 ],[3.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ],[1.0 ,0 ,0.0 ]])
    new_ =YO .uygula (P ,D ,[],[0.9 ,0.9 ],lambda X :np .ones (len (X )))
    assert np .allclose (new_ ,D )


def test_daha_iyi_skor_yonu_DEGISTIRIR ():
    P =np .array ([[0.0 ,0 ,0 ],[3.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ],[1.0 ,0 ,0.0 ]])
    new_ =YO .uygula (P ,D ,[],[0.9 ,0.9 ],
    lambda X :np .arange (len (X ),dtype =float ))
    assert not np .allclose (new_ [0 ],D [0 ])


def test_ciktilar_birim_vektor ():
    P =np .array ([[0.0 ,0 ,0 ],[3.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,2.0 ],[5.0 ,0 ,0.0 ]])
    new_ =YO .uygula (P ,D ,[sil ()],[0.5 ,0.5 ],
    lambda X :np .arange (len (X ),dtype =float ))
    assert np .allclose (np .linalg .norm (new_ ,axis =1 ),1.0 )


def test_bos_havuz_ve_tek_aday_PATLAMAZ ():
    assert len (YO .uygula (np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),[],[],lambda X :[]))==0 
    P =np .array ([[0.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ]])
    assert np .allclose (YO .uygula (P ,D ,[],[0.5 ],lambda X :np .zeros (len (X ))),D )


def test_oznitelik_sutun_sayisi_sabit ():
    P =np .array ([[0.0 ,0 ,0 ],[3.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ],[1.0 ,0 ,0.0 ]])
    _ ,X =YO .options (P ,D ,0 ,[sil ()],0.5 ,YO .baskin_yon (D ))
    assert X .shape [1 ]==len (YO .OZ_AD )


def test_bozuk_eksen_ATLANIR ():
    P =np .array ([[0.0 ,0 ,0 ]])
    D =np .array ([[0.0 ,0 ,1.0 ]])
    V ,X =YO .options (P ,D ,0 ,[{"axis":[0 ,0 ,0 ],"center":[0 ,0 ,0 ]}],
    0.5 ,None )
    assert len (V )==1 and np .isfinite (X ).all ()


def test_baskin_yon_isaret_hizali ():
    D =np .array ([[0.0 ,0 ,1.0 ],[0.0 ,0 ,-1.0 ],[0.0 ,0 ,1.0 ]])
    b =YO .baskin_yon (D )
    assert abs (abs (float (b @np .array ([0.0 ,0 ,1.0 ])))-1 )<1e-6 


def test_destek_ozniteligi_ayni_yonlu_adaylari_sayar ():
    P =np .array ([[0.0 ,0 ,0 ],[3.0 ,0 ,0 ],[6.0 ,0 ,0 ]])
    D =np .tile (np .array ([[0.0 ,0 ,1.0 ]]),(3 ,1 ))
    _ ,X =YO .options (P ,D ,0 ,[],0.5 ,YO .baskin_yon (D ))
    assert X [0 ,YO .OZ_AD .index ("destek")]==pytest .approx (3.0 )
