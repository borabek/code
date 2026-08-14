# -*- coding: utf-8 -*-
"""POSE HEAD'i BUYUK ARTIKLAR ICIN YENIDEN EGIT — cevrimdisi secme turu.

DIAGNOSIS (Bolum 21.53): dagitilan pose head'in onerdigi most large yer
degistirme **2.87 mm**; kirpma sinirini (3 mm) HIC zorlamiyor. Yani
`maks_mm` atil, sinirlayan modelin kendisi.

IKI OLASI MEKANIZMA -- ikisi de here test ediliyor:

 (1) SECIM YANLILIGI: training verisi only `tt = max(3, 0.06*diag)`
     inside ESLESMIS adaylardan kuruluyordu. -> `Q3_TOL=15` with yeniden
     kuruldu: large residual orani %10.2 -> **%18.4**, maks 12.1 -> 14.9 mm.

 (2) REGRESYON BUZULMESI: ormanin yaprak ortalamasi ucdegerleri iceri
     ceker. Taban veride ZATEN %10.2 large hedef vardi but model 2.87mm'yi
     asmiyor -- i.e. buzulme single basina yeterli aciklama may be.
     -> `min_samples_leaf` kucultulerek test ediliyor.

Bu betik EGITIP OLCMEZ; adaylari own between **grup-disi (OOF)**
karsilastirir and only kazanani zincire sokmaya value kilar. Olcut,
metrigin kendisi not but metrikle DOGRUDAN baglantili: kabul kutusuna
(2 mm) giren candidate orani.
"""
import json 
import os 
import pickle 
import sys 

import numpy as np 
from sklearn .ensemble import RandomForestRegressor 
from sklearn .model_selection import GroupKFold 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

N_FEAT =58 # dagitilan wire_gate.feats_for genisligi
KATLAR =4 


def yukle (fp ):
    d =np .load (fp ,allow_pickle =True )
    X =np .asarray (d ["X"],float )[:,:N_FEAT ]
    Y =np .asarray (d ["Y"],float )
    g =np .array ([str (x )for x in d ["geo"]])
    p =np .array ([str (x )for x in d ["pid"]])
    return X ,Y ,g ,p 


def val_gruplari ():
    """VAL parcalarinin GEOMETRI GRUPLARI -- egitimden cikarilacak."""
    try :
        gk =json .load (open ("results/_strict_geometry_keys.json",
        encoding ="utf-8"))
        s3 =json .load (open ("results/split3.json",encoding ="utf-8"))
    except Exception as e :# noqa: BLE001
        print (f"  UYARI: leakage kapisi kurulamadi ({type (e ).__name__ })")
        return set ()
        # NOTE: split3.json'da `val` a SOZLUK ({"n":..., "parts":[...]}).
        # Dogrudan on donmek anahtarlari ('n','parts') gives and leakage
        # kapisi SESSIZCE no seyi elemez -- first kosuda full boyle became
        # (0 grup cikarildi).
    v =s3 .get ("val")or s3 .get ("VAL")or []
    if isinstance (v ,dict ):
        v =v .get ("parts")or v .get ("pids")or []
    val =[str (x )for x in v ]
    grup ={gk [p ]for p in val if p in gk }
    assert val ,"VAL part listesi BOS -- leakage kapisi kurulamadi"
    assert grup ,"VAL parcalarinin hicbiri geometri haritasinda YOK"
    print (f"  VAL {len (val )} part -> {len (grup )} geometri grubu")
    return grup 


def oof_degerlendir (X ,Y ,g ,ad ,Xd =None ,Yd =None ,gd =None ,**kw ):
    """grup-disi prediction -> lateral artigin kabul kutusuna girme orani.

    ORTAK DEGERLENDIRME POPULASYONU (2026-08-14 duzeltmesi): training verisi
    (X,Y,g) with DEGERLENDIRME verisi (Xd,Yd,gd) ayrilir. Ilk kosuda two
    candidate KENDI data kumesinde puanlandi and genis data "more kotu" gorundu
    -- oysa genis cluster DAHA ZOR satirlar iceriyor. Ayni satirlarda
    olculmeyen two ratio KIYASLANAMAZ.
    """
    if Xd is None :
        Xd ,Yd ,gd =X ,Y ,g 
    tah =np .zeros_like (Yd )
    gkf =GroupKFold (n_splits =KATLAR )
    # Degerlendirme kumesini katlara bol; each fold for EGITIM kumesinden
    # that katin gruplarini CIKARARAK egit (grup-disi, sizintisiz).
    for _ ,te in gkf .split (Xd ,Yd ,groups =gd ):
        tut =~np .isin (g ,np .unique (gd [te ]))
        m =RandomForestRegressor (n_jobs =-1 ,random_state =0 ,**kw ).fit (
        X [tut ],Y [tut ])
        tah [te ]=m .predict (Xd [te ])
        # lateral residual: duzeltmeden ONCE |Y|, duzeltmeden SONRA |Y - prediction|
    onc =np .linalg .norm (Yd [:,:2 ],axis =1 )
    son =np .linalg .norm (Yd [:,:2 ]-tah [:,:2 ],axis =1 )
    oner =np .linalg .norm (tah [:,:2 ],axis =1 )
    print (f"{ad :34s} kutuda(<=2mm) {100 *(onc <=2 ).mean ():5.1f}% -> "
    f"{100 *(son <=2 ).mean ():5.1f}%  | artik ortanca "
    f"{np .median (onc ):.2f} -> {np .median (son ):.2f} mm | "
    f"oneri maks {oner .max ():5.2f} >3mm {100 *(oner >3 ).mean ():4.1f}%")
    return {"kutu_once":float ((onc <=2 ).mean ()),
    "kutu_sonra":float ((son <=2 ).mean ()),
    "artik_ortanca":float (np .median (son )),
    "oneri_maks":float (oner .max ()),
    "oneri_buyuk_oran":float ((oner >3 ).mean ())}


def main ():
    VG =val_gruplari ()
    print (f"leakage kapisi: {len (VG )} VAL geometri grubu egitimden CIKARILDI\n")
    kaynak =[("TABAN veri (tol ~3-6mm)","results/pose_veri_graf.npz"),
    ("GENIS veri (tol 15mm)","results/pose_veri_tol15.npz")]
    ayar =[("orman yaprak>=5 (MEVCUT)",dict (n_estimators =400 ,
    min_samples_leaf =5 )),
    ("orman yaprak>=2",dict (n_estimators =400 ,min_samples_leaf =2 )),
    ("orman yaprak>=1",dict (n_estimators =400 ,min_samples_leaf =1 ))]
    # ORTAK DEGERLENDIRME KUMESI: each candidate AYNI satirlarda puanlanir.
    # Taban data secildi because dagitilan modelin gordugu populasyon odur.
    Xd ,Yd ,gd ,_ =yukle (kaynak [0 ][1 ])
    _t =~np .isin (gd ,list (VG ))
    Xd ,Yd ,gd =Xd [_t ],Yd [_t ],gd [_t ]
    print (f"ORTAK DEGERLENDIRME KUMESI: {len (Yd )} satir / "
    f"{len (set (gd ))} grup\n")

    rapor ={}
    for vad ,vfp in kaynak :
        if not os .path .exists (vfp ):
            print (f"EKSIK: {vfp }")
            continue 
        X ,Y ,g ,p =yukle (vfp )
        tut =~np .isin (g ,list (VG ))
        X ,Y ,g =X [tut ],Y [tut ],g [tut ]
        print (f"--- EGITIM: {vad } -> {len (Y )} satir / {len (set (g ))} grup ---")
        for aad ,kw in ayar :
            rapor [f"{vad } | {aad }"]=oof_degerlendir (
            X ,Y ,g ,"  "+aad ,Xd =Xd ,Yd =Yd ,gd =gd ,**kw )
        print ()

    json .dump (rapor ,open ("results/pose_yeniden.json","w"),indent =1 )
    print ("-> results/pose_yeniden.json")

    # EN IYI ADAYI EGIT VE KAYDET (zincire sokulmak so as to)
    en =max (rapor ,key =lambda k :rapor [k ]["kutu_sonra"])
    print (f"\nEN IYI: {en }  (kutuda {100 *rapor [en ]['kutu_sonra']:.1f}%)")
    vad ,aad =[x .strip ()for x in en .split ("|")]
    vfp =dict (kaynak )[vad ]
    kw =dict (ayar )[aad ]
    X ,Y ,g ,p =yukle (vfp )
    tut =~np .isin (g ,list (VG ))
    m =RandomForestRegressor (n_jobs =-1 ,random_state =0 ,**kw ).fit (
    X [tut ],Y [tut ])
    cik ={"model":m ,"n_feat":N_FEAT ,"maks_mm":10.0 ,"direction":False ,
    "hedef":"[w_perp.u, w_perp.v, g.u, g.v] -- YEREL cercevede",
    "note":(f"2026-08-14 YENIDEN EGITIM. Kaynak={vad }, ayar={aad }. "
    f"VAL geometri gruplari ({len (VG )}) egitimden CIKARILDI. "
    f"maks_mm 3->10 (Bolum 21.53: eski model 2.87mm'yi "
    f"asmiyordu). OOF kutuda-ratio "
    f"{100 *rapor [en ]['kutu_sonra']:.1f}%.")}
    with open ("results/pose_head_yeni.pkl","wb")as fh :
        pickle .dump (cik ,fh )
    print ("-> results/pose_head_yeni.pkl (DAGITILMADI, measurement icin)")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
