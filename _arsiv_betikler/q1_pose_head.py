# -*- coding: utf-8 -*-
"""Q1: POST-GATE POSE HEAD -- robot-hazir for ASIL kaldirac.

TAVAN OLCUMU (results/t_tavan.json) net says:
    kahin GATE      robot 0.4500 -> 0.5091   (+0.059)  <- gate'in tasidigi
    + kahin YON              -> 0.6124   (+0.103)
    + kahin KONUM            -> 0.9372   (+0.325)  <- ASIL kaldirac
Yani gate ne up to iyilesirse iyilessin robota few yansiyor; is KONUM and YON'de.

NE YAPIYORUZ: gate KARARINDAN SONRA, kabul edilen each candidate for
    (a) YANAL ARTIK: adayin real opening eksenine dik sapmasi (3 bilesen, yerel cercevede)
    (b) YON DUZELTMESI: eksenin correct yone according to angle hatasi (2 bilesen, teget duzlemde)
prediction edilir and point/direction that up to duzeltilir.

WHY LEARNED, RULE DEGIL: this gece RULE tabanli four konum/direction kolu was tried and DORDU DE became
(izdusum uctan uca -0.045, axis uzlasisi net -110, yarik yonu net -187, B-rep kapisi 0.000).
Hepsinin ortak kusuru ayniydi: rule HERKESE uygulaniyordu. Ogrenilmis a kafa "ne up to and
hangi yone" sorusunu ADAY BASINA cevaplar and emin degilse SIFIR correction gives.

VERI: manufacturer GT'si (new insan etiketi YOK). Egitim only TESPITTE ESLESEN candidates on
(because residual however correct aciklikla eslesmis a candidate for tanimli).

SAFETY: correction MAKS_MM with sinirli; model belirsizse (prediction buyuklugu kucukse) dokunmaz.
Bu, "kararsizsa dokunma" ilkesinin somut hali.

OZELLIKLER (calisma aninda hesaplanabilir, GT'ye BAKMAZ): 58 gate sutunu + yerel cerceve
istatistikleri. Hedef YEREL CERCEVEDE ifade edilir (axis and two dik direction), so dunya
koordinatlarina bagimli olmaz -- part donerse de gecerli.

KILL (onceden yazili): robot-hazir >= +0.02 VE tespit kaybi < 0.005, and GA with kanitli.
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
DERZ ="results/_der_zengin.pkl"
MAKS_MM =3.0 # correction upper siniri (denetim tavsiyesi: 2-3mm)
MAKS_DEG =25.0 # direction duzeltmesi upper siniri


def local_frame (d ):
    """Eksen + two dik unit vektor (saga-elli). Hedefi dunya not YEREL cercevede ifade eder."""
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (d @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (d ,a );u /=np .linalg .norm (u )+1e-9 
    v =np .cross (d ,u )
    return d ,u ,v 


def main ():
    import karar_olcutu 
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestRegressor 
    from sklearn .model_selection import GroupKFold 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster (DERZ )
    measure_set .rapor_bas (rap )
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")

    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    DON =dag .get ("donusum")
    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tpid =np .array ([str (x )for x in zen ["pids"]])
    gk =measure_set .geo_anahtarlari ()
    tgrp =np .array ([gk .get (p ,"yok:"+p )for p in tpid ])
    tg ={r ["geo"]for r in DER }
    keep =~np .isin (tgrp ,list (tg ))

    def donustur (X ,pidler ):
        if not DON :
            return X 
        Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
        for u in np .unique (pidler ):
            i =np .where (pidler ==u )[0 ]
            Z [i ]=wire_gate .within_part (X [i ],DON )
        return Z 

    from sklearn .ensemble import RandomForestClassifier 
    Mt =donustur (Xt ,tpid )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Mt [keep ],ytr [keep ])
    gate ={"clf":clf ,"n_feat":Mt .shape [1 ],"donusum":DON }
    print (f"gate hazir ({Mt .shape [1 ]} sutun, sizintisiz)",flush =True )

    # --- POSE EGITIM VERISI: only TESPITTE ESLESEN candidates
    RX ,RY ,RG ,RPID =[],[],[],[]
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None or not len (r ["G"]):
            continue 
        P =r ["P"];Pd =r ["Pd"]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe_t =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *float (r ["diag"]))
        used ,hit =set (),set ()
        X58 =np .hstack ([r ["X"],r ["XR"]])
        for dd ,a_ ,b_ in sorted ((pe_t [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if dd >tt or a_ in used or b_ in hit :
                continue 
            used .add (a_ );hit .add (b_ )
            d ,u ,v =local_frame (Pd [a_ ])
            # TARGET 1-2: GT axis dogrusuna which is sapmanin YEREL bilesenleri (u, v)
            w =G [b_ ]-P [a_ ]
            w_perp =w -float (w @Gd [b_ ])*Gd [b_ ]# GT eksenine dik bilesen
            # TARGET 3-4: direction duzeltmesi (GT ekseninin yerel cercevedeki teget bilesenleri)
            g =Gd [b_ ]*(1.0 if float (Gd [b_ ]@d )>=0 else -1.0 )
            RX .append (X58 [a_ ])
            RY .append ([float (w_perp @u ),float (w_perp @v ),float (g @u ),float (g @v )])
            RG .append (r ["geo"]);RPID .append (r ["pid"])
    RX =np .array (RX ,float );RY =np .array (RY ,float );RG =np .array (RG )
    print (f"pose training: {len (RY )} eslesen candidate | lateral |w| medyan "
    f"{np .median (np .linalg .norm (RY [:,:2 ],axis =1 )):.2f}mm | "
    f"direction deviation medyan {np .degrees (np .arcsin (np .clip (np .linalg .norm (RY [:,2 :],axis =1 ),0 ,1 ))).mean ():.1f} deg",
    flush =True )

    # OOF pose tahmini (geometri grubuna according to)
    oof =np .zeros_like (RY )
    for tr ,te in GroupKFold (n_splits =5 ).split (RX ,RY [:,0 ],RG ):
        m =RandomForestRegressor (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (RX [tr ],RY [tr ])
        oof [te ]=m .predict (RX [te ])
    err0 =np .linalg .norm (RY [:,:2 ],axis =1 )
    err1 =np .linalg .norm (RY [:,:2 ]-oof [:,:2 ],axis =1 )
    print (f"YANAL artik: once medyan {np .median (err0 ):.2f}mm -> sonra {np .median (err1 ):.2f}mm | "
    f"<=2mm orani {np .mean (err0 <=2 ):.1%} -> {np .mean (err1 <=2 ):.1%}")
    a0 =np .degrees (np .arcsin (np .clip (np .linalg .norm (RY [:,2 :],axis =1 ),0 ,1 )))
    a1 =np .degrees (np .arcsin (np .clip (np .linalg .norm (RY [:,2 :]-oof [:,2 :],axis =1 ),0 ,1 )))
    print (f"YON      : once medyan {np .median (a0 ):.2f} deg -> sonra {np .median (a1 ):.2f} deg | "
    f"<=10 deg orani {np .mean (a0 <=10 ):.1%} -> {np .mean (a1 <=10 ):.1%}")

    with io .open ("results/q1_pose_head.json","w",encoding ="utf-8")as f :
        json .dump ({"n":int (len (RY )),
        "yanal_once_med":float (np .median (err0 )),"yanal_sonra_med":float (np .median (err1 )),
        "yanal_2mm_once":float (np .mean (err0 <=2 )),"yanal_2mm_sonra":float (np .mean (err1 <=2 )),
        "yon_once_med":float (np .median (a0 )),"yon_sonra_med":float (np .median (a1 )),
        "yon_10_once":float (np .mean (a0 <=10 )),"yon_10_sonra":float (np .mean (a1 <=10 ))},
        f ,indent =1 )
    print ("receipt -> results/q1_pose_head.json")
    with open ("results/q1_pose_veri.pkl","wb")as f :
        pickle .dump ({"RX":RX ,"RY":RY ,"RG":RG ,"RPID":np .array (RPID ),"oof":oof },f )


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
