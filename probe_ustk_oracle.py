# -*- coding: utf-8 -*-
"""BELIRLEYICI AYRISTIRMA: SIRALAMA mi, ADET/ESIK mi?

PARADOKS. NIT'te pool GT'nin %89.8'ini tasiyor, DOGRU option 7470 option
between first ~40'ta (order yuzdeligi 0.005) -- but uctan uca F1 = 0.005.
Siralama iyi, pool iyi, sonuc SIFIR.

TEK ACIKLAMA ADAYI: part basina 24.4 CP secilmesi is required and ESIK TABANLI
rule bunu yapamiyor. Skor ayrimi 0.05 iken threshold ya yuzlerce sey aliyor
(precision cokuyor) ya da a avuc (recall cokuyor).

BU SONDA UC KOLU AYNI SKORLA kiyaslar:
  1. RULE      : bugunku secim (goreli/mutlak threshold + NMS)
  2. USTK_KAHIN : same skorla first k, k = PARCANIN GERCEK CP SAYISI
  3. TAVAN      : mukemmel selector (havuzda ulasilabilen each GT)

YORUM:
  USTK_KAHIN >> RULE  -> sorun ADET/ESIK. Adet tahmini + yapisal uretim
                          (lattice) kolu ACILIR; ranking already yeterli.
  USTK_KAHIN ~ RULE   -> sorun SIRALAMA. Adet bilgisi kurtarmaz; secicinin
                          kendisi degismeli.

`k` GERCEK CP sayisidir -- this a KAHIN, urun not. Amac upper siniri gormek.
D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

KUME =os .environ .get ("UK_KUME","d6")
KAT_MIN =int (os .environ .get ("UK_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
NMS =5.0 
KURAL =("goreli",0.85 ,0.20 )
EKSEN_TOL =40.0 


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def _tavan (d ):
    """Havuzda kabul kutusuna entering GT count (mukemmel selector)."""
    P =d ["P"][d ["idx"]]
    YD =d ["YD"]
    G =np .asarray (d ["G"],float )
    Gd =np .asarray (d ["Gd"],float )
    if not len (P )or not len (G ):
        return 0 
    Gd =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
    v =P [:,None ,:]-G [None ,:,:]
    al =(v *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (v -al [...,None ]*Gd [None ,:,:],axis =-1 )
    an =np .degrees (np .arccos (np .clip (YD @Gd .T ,-1 ,1 )))
    ok =(pe <=K .YANAL )&(np .abs (al )<=EKSEN_TOL )&(an <=K .ACI )
    return int (ok .any (0 ).sum ())


def _ustk (d ,s ,k ):
    """Skora according to first k option, NMS with (same kabul kutusu)."""
    P =d ["P"][d ["idx"]]
    YD =d ["YD"]
    rank_ =np .argsort (-np .asarray (s ))
    secP ,secD =[],[]
    for j in rank_ :
        if len (secP )>=k :
            break 
        p =P [j ]
        if secP and min (np .linalg .norm (np .asarray (secP )-p ,axis =1 ))<NMS :
            continue 
        secP .append (p )
        secD .append (YD [j ])
    return (np .asarray (secP ).reshape (-1 ,3 ),np .asarray (secD ).reshape (-1 ,3 ))


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar } ({time .time ()-t0 :.0f} s)",
    flush =True )

    oof =[None ]*len (data_ )
    for b in katlar :
        ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
        n_satir =sum (len (data_ [i ]["y"])for i in ic )
        M =np .empty ((n_satir ,data_ [0 ]["_M"].shape [1 ]),np .float32 )
        o =0 
        for i in ic :
            m_ =data_ [i ]["_M"]
            M [o :o +len (m_ )]=m_ 
            o +=len (m_ )
        Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
        rng =np .random .default_rng (0 )
        poz =np .where (Y ==1 )[0 ]
        neg =np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        m =HistGradientBoostingClassifier (
        max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
        del M 
        for i in dis :
            oof [i ]=m .predict_proba (data_ [i ]["_M"])[:,1 ]
        print (f"  OOF {b } ({time .time ()-t0 :.0f} s)",flush =True )

    agg =collections .defaultdict (lambda :collections .Counter ())
    for d ,s in zip (data_ ,oof ):
        if s is None :
            continue 
        a =agg [d ["mfg"]]
        G ,Gd ,dg =np .asarray (d ["G"],float ),np .asarray (d ["Gd"],float ),d ["diag"]
        a ["gt"]+=len (G )
        a ["ceiling"]+=_tavan (d )
        # 1) BUGUNKU RULE
        P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,KURAL ,nms_mm =NMS )
        tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
        signed =True )[:3 ]
        a ["k_tp"]+=tp ;a ["k_fp"]+=fp ;a ["k_fn"]+=fn 
        # 2) USTK KAHIN (k = real CP count)
        P2 ,D2 =_ustk (d ,s ,len (G ))
        tp2 ,fp2 ,fn2 =match_hungarian (P2 ,D2 ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
        signed =True )[:3 ]
        a ["u_tp"]+=tp2 ;a ["u_fp"]+=fp2 ;a ["u_fn"]+=fn2 

    def f1 (tp ,fp ,fn ):
        return 2 *tp /max (2 *tp +fp +fn ,1 )

    print (f"\n{'brand':<7}{'GT':>7}{'KURAL':>9}{'USTK_KAHIN':>12}"
    f"{'TAVAN':>9}{'kazanc':>9}")
    out ,T ={},collections .Counter ()
    for m_ in sorted (agg ,key =lambda x :-agg [x ]["gt"]):
        a =agg [m_ ]
        for k_ in a :
            T [k_ ]+=a [k_ ]
        kf =f1 (a ["k_tp"],a ["k_fp"],a ["k_fn"])
        uf =f1 (a ["u_tp"],a ["u_fp"],a ["u_fn"])
        tv =a ["ceiling"]/max (a ["gt"],1 )
        tv =2 *tv /(1 +tv )
        out [m_ ]={"gt":a ["gt"],"rule":kf ,"ustk_kahin":uf ,"ceiling":tv }
        print (f"{m_ :<7}{a ['gt']:>7}{kf :>9.4f}{uf :>12.4f}{tv :>9.4f}"
        f"{uf -kf :>+9.4f}")
    kf =f1 (T ["k_tp"],T ["k_fp"],T ["k_fn"])
    uf =f1 (T ["u_tp"],T ["u_fp"],T ["u_fn"])
    tv =T ["ceiling"]/max (T ["gt"],1 )
    tv =2 *tv /(1 +tv )
    print (f"{'TOPLAM':<7}{T ['gt']:>7}{kf :>9.4f}{uf :>12.4f}{tv :>9.4f}"
    f"{uf -kf :>+9.4f}")
    print (f"\nYORUM: USTK_KAHIN - KURAL = {uf -kf :+.4f}")
    print ("  BUYUKSE -> sorun ADET/ESIK (lattice + count tahmini kolu ACILIR)")
    print ("  ~0 ISE  -> sorun SIRALAMA (secicinin kendisi degismeli)")
    json .dump ({"dizin":os .environ ["P6_DIZIN"],"cluster":KUME ,"brand":out ,
    "total":{"rule":kf ,"ustk_kahin":uf ,"ceiling":tv ,
    "gt":T ["gt"]},
    "not":"USTK_KAHIN: same skorla ilk k, k = GERCEK CP sayisi. "
    "KAHIN'dir, urun not. Amac SIRALAMA with ADET/ESIK "
    "sorununu ayirmak. D7'ye BAKILMADI."},
    open (f"results/ustk_kahin_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/ustk_kahin_{KUME }.json")


if __name__ =="__main__":
    main ()
