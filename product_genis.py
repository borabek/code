# -*- coding: utf-8 -*-
"""GENISLETILMIS URUN YOLU: B-rep havuzu + mouth tanimlayicilari + sign correction.

MEASURED (D7 = 835 part brand-disi, TAM ZINCIR, MIKRO, threshold D6'da secildi):
| yigin | robot | tespit | makro |
|---|---|---|---|
| dagitilan urun (v6 + NMS) | 0.2029 | 0.4523 | 0.2146 |
| **this path** | **0.3090** | **0.4813** | **0.3142** |

10/12 markada artis; real loss YOK (CCD -0.007 duz, C3 -0.009 five parcada).
Makbuzlar: `results/secici_ailesi.json`, `results/b2a_isaret.json`,
`results/tam_havuz_gate.json`.

UC KALDIRAC (all of them single single measured):
 1. ETIKET TANIMI duzeltmesi -- gate korpusu `build_zengin_parite.py`'nin
    tanimiyla (lateral + axial 40mm + acgozlu bire-a). Tek basina 0.1159 -> 0.2009.
 2. B-REP HAVUZU (silindir agizlari + duzlemsel opening merkezleri, 3mm dedupe):
    0.2213 -> 0.2563. Mesh tepeleri EKLENMEZ -- three bagimsiz olcumde ZARAR verdi.
 3. AGIZ TANIMLAYICILARI (9 column) + HGB-derin + ISARET DUZELTME: -> 0.3090.

TEZE SADIK: DiffusionNet 5 sinif, ~6000 uniform izotropik remesh and `v_o`
mouth-ortasi turetmesi DEGISMEDI. B-rep onerileri tezin adaylarinin YANINA
eklenen ikinci a kaynaktir; tezin cevabi always havuzda and `source==0`
with isaretlidir. Sonuclar "tez sonucu" not "tez-omurgali genisletme" as
raporlanir.

KAPATMA: `cp_config.json` inside `robot_genis_havuz=false` ya da
`URUN_GENIS=0` cevre degiskeni -> urun old yoluna returns.
"""
import os 

import numpy as np 

import mouth_descriptor 
import brep_pool 
import wire_gate 

GIRME =mouth_descriptor .AD .index ("girme")
ERISIM =mouth_descriptor .AD .index ("erisim")


def _cfg (ad ,cevre ,vars_ ):
    v =os .environ .get (cevre )
    if v is not None :
        return v not in ("0","","false","False")
    try :
        import json 
        return bool (json .load (open ("cp_config.json",encoding ="utf-8"))
        .get (ad ,vars_ ))
    except Exception :
        return vars_ 


ACIK =_cfg ("robot_genis_havuz","URUN_GENIS",True )
ESIK =0.05 # D6'da secildi (`results/secici_ailesi.json`), D7'de taranmadi


def pool (P_seg ,D_seg ,cyl ,acik ):
    """Tezin `v_o` adaylari + B-rep agizlari. Mesh tepeleri KASTEN YOK.

    Mesh tepeleri pool recall'unu 0.6654 -> 0.8465 acar but uctan uca UC
    bagimsiz olcumde ZARAR verdi (0.2972 vs 0.3095): FP neredeyse ikiye
    katlaniyor. Tavan acmak yetmiyor, selector kullanamiyor.
    """
    return brep_pool .merged_pool (P_seg ,D_seg ,cyl ,acik )


def tanimlayici (P ,D ,cyl ,mesh ,diag ):
    """Agiz olculeri. `cyl` otherwise only isin olculeri dolar."""
    met =[]
    if cyl :
        C =np .asarray ([c ["center"]for c in cyl ],float )
        A =np .asarray ([c ["axis"]for c in cyl ],float )
        n =np .linalg .norm (A ,axis =1 ,keepdims =True )
        A =A /np .maximum (n ,1e-12 )
        for p in np .asarray (P ,float ).reshape (-1 ,3 ):
            w =p [None ]-C 
            e =np .einsum ("ij,ij->i",w ,A )
            d =np .linalg .norm (w -e [:,None ]*A ,axis =1 )
            i =int (np .argmin (d ))
            m ={"radius":float (cyl [i ].get ("radius",0.0 ))}
            if cyl [i ].get ("mouth_a")is not None :
                m ["mouth_a"]=cyl [i ]["mouth_a"]
                m ["mouth_b"]=cyl [i ].get ("mouth_b")
            met .append (m )
    else :
        met =[{}for _ in range (len (P ))]
    return mouth_descriptor .tanimla (P ,D ,met ,mesh ,diag )


def isaret_duzelt (D ,T ):
    """Tel DISARIDAN girer: disari yolu iceriden kisaysa direction TERS cevrilir.

    Olculdu (`results/b2a_isaret.json`): +0.0316 robot, tespit DEGISMEDI.
    Saf fiziksel rule ogrenilmis siniflandiricinin %94'unu veriyor; urunde
    OGRENME YOK, rule present -- more few hareketli part.
    """
    D =np .asarray (D ,float ).reshape (-1 ,3 )
    T =np .asarray (T ,float )
    if not len (D ):
        return D 
    return np .where ((T [:,ERISIM ]<T [:,GIRME ])[:,None ],-D ,D )


def sec (P ,D ,X58 ,cyl ,mesh ,diag ,model ,threshold =ESIK ):
    """Genisletilmis yolun DECISION fonksiyonu. Doner: (P, D) secilmis candidates.

    Sira: feature -> gate -> threshold -> NMS -> ISARET DUZELTME.
    (Poz kafasi this fonksiyonun DISINDA, urun zincirinde kalir.)
    """
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    D =np .asarray (D ,float ).reshape (-1 ,3 )
    if len (P )<2 :
        return P ,D 
    T =tanimlayici (P ,D ,cyl ,mesh ,diag )
    X =np .hstack ([np .asarray (X58 ,float ),T ])
    s =np .asarray (model .predict_proba (
    wire_gate .within_part (X ,"zskor"))[:,1 ],float )
    k =s >=threshold 
    if not k .any ():
        return P [:0 ],D [:0 ]
    P ,D ,T ,s =P [k ],D [k ],T [k ],s [k ]
    if len (P )>1 :
        nm =wire_gate .crowd_mask (P ,s )
        P ,D ,T =P [nm ],D [nm ],T [nm ]
    return P ,isaret_duzelt (D ,T )


MODEL_YOL ="results/kazanan_hgb_derin.pkl"
_MODEL =None 


def model_yukle (yol =MODEL_YOL ):
    """HGB-derin gate. Yoksa None returns -> cagiran ESKI yola duser."""
    global _MODEL 
    if _MODEL is None :
        if not os .path .exists (yol ):
            return None 
        import pickle 
        _MODEL =pickle .load (open (yol ,"rb"))["HGB-derin"]
    return _MODEL 


def brep_cikar (step_path ):
    """Silindir + duzlemsel opening. Cikarim aninda ~0.7 s/part.

    Onbellek YOK -- urun new a part gorur and STEP'ten hesaplamak zorundadir.
    STEP verilmezse (None) arm devre disi kalir and cagiran ESKI yola duser.
    """
    if not step_path or not os .path .exists (step_path ):
        return None ,None 
    import brep_aciklik 
    import brep_snap 
    try :
        cyl =brep_snap .exact_cylinders (step_path )
    except Exception :
        cyl =[]
    try :
        acik =brep_aciklik .acikliklar (step_path )
    except Exception :
        acik =[]
    return cyl ,acik 


def out_ (V ,F ,probs ,cps_seg ,step_path ,CE ,CT ):
    """URUNUN genisletilmis ciktisi. Doner: cps listesi (point/direction).

    Girdi `cps_seg` tezin `v_o` adaylaridir and HAVUZDA KALIR (source 0).
    Kol calisamiyorsa (model absent / STEP absent / B-rep empty) None returns and cagiran
    ESKI yola duser -- sessizce bozuk output URETILMEZ.
    """
    model =model_yukle ()
    if model is None or not cps_seg :
        return None 
    cyl ,acik =brep_cikar (step_path )
    if cyl is None :
        return None 
    import trimesh 
    Ps =np .asarray ([c ["point"]for c in cps_seg ],float )
    Ds =np .asarray ([c ["direction"]for c in cps_seg ],float )
    P ,D ,_kay =pool (Ps ,Ds ,cyl ,acik )
    if len (P )<2 :
        return None 
    diag =float (np .linalg .norm (np .asarray (V ).max (0 )-np .asarray (V ).min (0 )))
    mesh =trimesh .Trimesh (np .asarray (V ,float ),np .asarray (F ,np .int64 ),
    process =False )
    X58 =np .asarray (wire_gate .feats_for (
    V ,F ,probs ,[{"point":P [i ],"direction":D [i ]}
    for i in range (len (P ))],CE ,CT ,step_path =step_path ),
    float )
    P2 ,D2 =sec (P ,D ,X58 ,cyl ,mesh ,diag ,model )
    return [{"point":P2 [i ],"direction":D2 [i ],"wire_score":1.0 }
    for i in range (len (P2 ))]
