"""K1.7: yerel goreli radius ozellikleri + old ckpt width korumasi.

Ozellikler OPT-IN (P3C_YEREL_YARICAP=1); temiz A/B null output, dagitilmadi.
Bu file ozelliklerin DOGRU hesaplandigini and korumanin calistigini sinar.
"""
import os 
os .environ ["P3C_YEREL_YARICAP"]="1"
import numpy as np 
import pytest 

import p3c_axis_selector as P3C 


def _cyl (merkez ,axis ,yaricap ,boy =4.0 ):
    m =np .asarray (merkez ,float )
    e =np .asarray (axis ,float )
    e =e /np .linalg .norm (e )
    return {"axis":e ,"radius":yaricap ,
    "mouth_a":m -e *boy /2 ,"mouth_b":m +e *boy /2 }


def test_yeni_ozellikler_eklendi ():
    assert P3C .OZ_AD [-4 :]==["yaricap_orani","yaricap_sira",
    "yerel_en_buyuk","n_yerel"]


def test_yerel_en_buyuk_dogru_isaretlenir ():
    """Komsulukta 0.5 and 2.0mm two silindir: only 2.0 'most large' must be."""
    p =np .zeros (3 )
    d =np .array ([1.0 ,0.0 ,0.0 ])
    cyls =[_cyl ([0.5 ,0 ,0 ],[1 ,0 ,0 ],0.5 ),
    _cyl ([0.5 ,0.5 ,0 ],[0 ,1 ,0 ],2.0 )]
    opt =P3C .secenekler (cyls ,p ,d ,diag =50.0 ,gate_s =0.9 ,komsu =None ,n_aday =3 )
    i_oran =P3C .OZ_AD .index ("yaricap_orani")
    i_enb =P3C .OZ_AD .index ("yerel_en_buyuk")
    i_yar =P3C .OZ_AD .index ("yaricap")
    # mevcut (tez cevabi) haric secenekler
    tur =[(o [2 ][i_yar ],o [2 ][i_oran ],o [2 ][i_enb ])for o in opt [1 :]]
    assert tur ,"silindir secenegi uretilmedi"
    for yar ,ratio ,enb in tur :
        if abs (yar -2.0 )<1e-9 :
            assert enb ==1.0 and abs (ratio -1.0 )<1e-9 
        else :
            assert enb ==0.0 and ratio <1.0 


def test_tez_cevabi_her_zaman_ilk_secenek ():
    """Tez sadakati: v_o turetmesi havuzdan DUSURULEMEZ."""
    p =np .array ([1.0 ,2.0 ,3.0 ])
    d =np .array ([0.0 ,0.0 ,1.0 ])
    opt =P3C .secenekler ([_cyl ([1.2 ,2 ,3 ],[1 ,0 ,0 ],2.0 )],p ,d ,
    diag =50.0 ,gate_s =0.5 ,komsu =None ,n_aday =1 )
    assert np .allclose (opt [0 ][0 ],p )and np .allclose (opt [0 ][1 ],d )
    assert opt [0 ][2 ][P3C .OZ_AD .index ("mevcut_mu")]==1.0 


def test_oz_uzunlugu_oz_ad_ile_ayni ():
    opt =P3C .secenekler ([_cyl ([0.5 ,0 ,0 ],[1 ,0 ,0 ],1.0 )],np .zeros (3 ),
    np .array ([1.0 ,0 ,0 ]),50.0 ,0.5 ,None ,1 )
    for o in opt :
        assert len (o [2 ])==len (P3C .OZ_AD )


class _Sahte :
    """13 feature bekleyen old ckpt taklidi."""
    n_features_in_ =13 


def test_eski_ckpt_genisligi_daraltilir ():
    X =np .arange (17 *2 ,dtype =float ).reshape (2 ,17 )
    Y =P3C ._uyumlu (_Sahte (),X )
    assert Y .shape ==(2 ,13 )
    # new ozellikler SONA eklendi -> first 13 column degismemeli
    assert np .array_equal (Y ,X [:,:13 ])


def test_dar_girdi_SESSIZCE_gecmez ():
    """Model more genis bekliyorsa patlamali; silent wrong skor URETMEMELI."""
    class _Genis :
        n_features_in_ =20 
    with pytest .raises (ValueError ,match ="refit"):
        P3C ._uyumlu (_Genis (),np .zeros ((2 ,17 )))
