# -*- coding: utf-8 -*-
"""A1+A2+A3: OLCUM CERRAHISI -- axial tolerans, ambiguous eslesme, angle gorunurlugu.

OTOPSI BULGUSU (2026-08-04): metrik three yerden kordu.
  A1  EKSENEL TOLERANS 40mm -- kodda gomulu, gerekcesiz. Olculen seat->mouth farki
      medyan 7.5-10.6mm (some parcalarda 0.0). 40mm that farkin ~4 kati.
  A2  BELIRSIZ ESLESME -- GT'lerin %29.3'unun kabul kutusunda BASKA a GT present
      (low-CP %20.8, very-CP %33.0). "Dogru" counted eslesme komsu delige ait may be.
  A3  ACI -- detection olcutu aciya TAMAMEN KOR (am=180). 180 derece TERS a CP tabloda
      "correct" gorunuyor.

BU BETIK HICBIR SEYI DAGITMAZ. Dort sayiyi YAN YANA koyar:
      F1(axis 40)      = bugunku headline (sisik)
      F1(axis 15)      = axial toleransi olcuye dayandirilmis hali
      F1_kesin(15)      = ambiguous eslesmeler KREDILENDIRILMEZ (lower boundary)
      + angle dagilimi
MANSET DUSECEK. Bu a loss not, sisintinin geri alinmasidir.
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import wire_gate 
from sina_cluster import match_greedy ,f1w ,f1_rejim 


def urun_ciktisi (r ,gate ):
    """Urunun karar yolu -> (P, Pd). tezgah2.puanla with AYNI zincir."""
    P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
    if r ["X"]is not None and r .get ("XR")is not None :
        X =np .hstack ([r ["X"],r ["XR"]])
        if X .shape [1 ]*2 ==gate ["n_feat"]:
            k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X ))
            if k .any ():
                P =np .asarray (r ["P"],float )[k ].copy ()
                Pd =np .asarray (r ["Pd"],float )[k ].copy ()
                c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                c =wire_gate .pose_correct (X [k ],c )
                c =wire_gate .angle_correct (X [k ],c )
                if r .get ("UYE"):
                    c =wire_gate .pick_member_direction (X [k ],c ,r ["UYE"])
                P =np .array ([x ["point"]for x in c ],float )
                Pd =np .array ([x ["direction"]for x in c ],float )
    return P ,Pd 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    DER ,gate ,ek =T2 .yukle ()

    CIKTI ={}
    for r in DER :
        CIKTI [r ["pid"]]=urun_ciktisi (r ,gate )

    def olc (eksen_tol ,kesin =False ,robot =False ):
        """conclusive=True: ambiguous eslesmeler TP sayilmaz (lower boundary)."""
        rows ,aci ,belirsiz ,total_ =[],[],0 ,0 
        for r in DER :
            P ,Pd =CIKTI [r ["pid"]]
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="very"if r ["n"]>=8 else "low"
            if robot :
                tp ,fp ,fn ,b =match_greedy (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,
                signed =True ,eksen_tol =eksen_tol )
            else :
                tp ,fp ,fn ,b =match_greedy (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ,
                eksen_tol =eksen_tol )
            nb =b ["ambiguous"];belirsiz +=nb ;total_ +=tp 
            if kesin and nb :
                tp -=nb ;fp +=nb ;fn +=nb 
            rows .append ((rj ,tp ,fp ,fn ))
            aci +=[e [4 ]for e in b ["eslesme"]]
        return rows ,np .array (aci ),belirsiz ,total_ 

    print ("="*78 )
    print ("A1 -- EKSENEL TOLERANS (detection)")
    print ("="*78 )
    R ={}
    for et in (40.0 ,25.0 ,15.0 ,10.0 ):
        rows ,aci ,bel ,top =olc (et )
        R [et ]={"f1":f1w (rows ),"ambiguous":bel ,"tp":top }
        rj =f1_rejim (rows )if callable (globals ().get ("f1_rejim"))else {}
        print (f"  eksen_tol {et :>5.1f}mm -> detection F1 {f1w (rows ):.4f} | TP {top } | "
        f"belirsiz {bel } (%{100 *bel /max (top ,1 ):.1f})")
    d =R [15.0 ]["f1"]-R [40.0 ]["f1"]
    print (f"\n  40mm -> 15mm FARK: {d :+.4f}  (headline {R [40.0 ]['f1']:.4f} -> {R [15.0 ]['f1']:.4f})")

    print ("\n"+"="*78 )
    print ("A2 -- BELIRSIZ ESLESME (F1 vs F1_kesin)")
    print ("="*78 )
    for et in (40.0 ,15.0 ):
        r1 ,_ ,b1 ,t1 =olc (et ,kesin =False )
        r2 ,_ ,_ ,_ =olc (et ,kesin =True )
        print (f"  eksen_tol {et :>5.1f}mm  F1 {f1w (r1 ):.4f}  |  F1_kesin {f1w (r2 ):.4f}"
        f"  |  diff {f1w (r2 )-f1w (r1 ):+.4f}  (belirsiz {b1 }/{t1 })")

    print ("\n"+"="*78 )
    print ("A3 -- ACI GORUNURLUGU (detection eslesmelerinin ACI dagilimi, ISARETSIZ axis)")
    print ("="*78 )
    _ ,aci ,_ ,_ =olc (15.0 )
    if len (aci ):
        for q in (50 ,75 ,90 ,95 ,99 ):
            print (f"  %{q :<3} {np .percentile (aci ,q ):>7.1f} derece")
        for threshold in (10 ,30 ,60 ,90 ):
            print (f"  aci > {threshold :>2} derece which eslesme: {int ((aci >threshold ).sum ())}/{len (aci )}"
            f" (%{100 *(aci >threshold ).mean ():.1f})")

    print ("\n"+"="*78 )
    print ("ROBOT (FIZIKSEL) metrigi same cerrahi with")
    print ("="*78 )
    for et in (40.0 ,15.0 ):
        rows ,_ ,bel ,top =olc (et ,robot =True )
        rows2 ,_ ,_ ,_ =olc (et ,kesin =True ,robot =True )
        print (f"  eksen_tol {et :>5.1f}mm  robot {f1w (rows ):.4f} | kesin {f1w (rows2 ):.4f}"
        f" | belirsiz {bel }/{top }")

    with io .open ("results/a1a3_metric_cerrahi.json","w",encoding ="utf-8")as f :
        json .dump ({"detection":{str (k ):v for k ,v in R .items ()},
        "aci_p90":float (np .percentile (aci ,90 ))if len (aci )else None ,
        "aci_60_ustu":int ((aci >60 ).sum ())if len (aci )else 0 ,
        "not":"A1 axis 40->15, A2 ambiguous bayragi, A3 angle gorunurlugu"},
        f ,indent =1 ,ensure_ascii =False )
    print ("\nmakbuz -> results/a1a3_metric_cerrahi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
