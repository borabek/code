# -*- coding: utf-8 -*-
"""YENI-1 -- KAYIP FONKSIYONU: focal / sinif-dengeli, S7 teshisine dogrudan nisan

WHY. S7 cokusun sebebini olctu: POZ-NEG SKOR AYRIMI. UPUN'da 0.847,
NIT'te 0.050 -- i.e. NIT'te correct secenek yanlistan only 0.05 more
high score aliyor. Bugune up to denenen each sey OZNITELIK ekliyordu; loss
fonksiyonuna HIC dokunulmadi. Oysa ayrimi dogrudan sekillendiren sey odur.

DORT KOL (all of them AYNI features, AYNI katlar, AYNI rule aramasi):
  baseline        : bugunku HGB (lower-orneklenmis, duz log-loss)
  weight      : lower-ornekleme YOK, POZITIFE weight (class-balanced)
  focal        : easy negatifleri bastiran odak kaybi (gamma)
  weight+odak : ikisi birden

HGB'de dogrudan focal absent; `sample_weight` with YAKLASILIR:
  w_i = (1-p_i)^gamma   -- p_i a ON GECISTEN gelen olasilik
Yani before a baseline model, after onun easy buldugu orneklerin agirligi
dusurulerek IKINCI model. Bu, focal kaybin pratik karsiligidir.

KAPI: +0.01 (mikro robot F1, `full` brand katlari mantigi).
D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

import makbuz_hash 

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

KUME =os .environ .get ("KD_KUME","d6")
KAT_MIN =int (os .environ .get ("KD_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
GAMMA =float (os .environ .get ("KD_GAMMA","2.0"))
NMS =5.0 
KURALLAR =([("mutlak",e )for e in (0.20 ,0.40 ,0.60 ,0.80 ,0.90 ,0.95 )]+
[("goreli",o ,t )for o in (0.50 ,0.70 ,0.85 ,0.95 )
for t in (0.05 ,0.20 ,0.40 )])


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])]).astype (
    np .float32 )


def yap (**kw ):
    return HistGradientBoostingClassifier (
    max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
    l2_regularization =1.0 ,random_state =0 ,**kw )


def puanla (data_ ,skor ,rule_ ):
    per =collections .defaultdict (lambda :[0 ,0 ,0 ])
    for d ,s in zip (data_ ,skor ):
        P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,rule_ ,nms_mm =NMS )
        a ,b ,c =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,K .ACI ,
        False ,signed =True )[:3 ]
        q =per [d ["mfg"]]
        q [0 ]+=a ;q [1 ]+=b ;q [2 ]+=c 
    T =[sum (q [i ]for q in per .values ())for i in range (3 )]
    pm ={m :2 *q [0 ]/max (2 *q [0 ]+q [1 ]+q [2 ],1 )for m ,q in per .items ()}
    return {"robot":2 *T [0 ]/max (2 *T [0 ]+T [1 ]+T [2 ],1 ),
    "makro":float (np .mean (list (pm .values ())))if pm else 0.0 ,
    "TP":T [0 ],"FP":T [1 ],"FN":T [2 ]}


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar } | gamma={GAMMA }",flush =True )

    KOLLAR =("baseline","agirlik","focal","agirlik_focal")
    top ={k :collections .Counter ()for k in KOLLAR }
    kat_sonuc ={}
    for b in katlar :
        ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
        n_s =sum (len (data_ [i ]["y"])for i in ic )
        M =np .empty ((n_s ,data_ [0 ]["_M"].shape [1 ]),np .float32 )
        o =0 
        for i in ic :
            m_ =data_ [i ]["_M"]
            M [o :o +len (m_ )]=m_ 
            o +=len (m_ )
        Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
        rng =np .random .default_rng (0 )
        poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
        alt =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])

        modeller ={}
        # 1) TABAN: lower-ornekleme + duz loss
        modeller ["baseline"]=yap ().fit (M [alt ],Y [alt ])
        # 2) AGIRLIK: TUM data, pozitife sinif agirligi
        w =np .where (Y ==1 ,len (neg )/max (len (poz ),1 ),1.0 )
        modeller ["agirlik"]=yap ().fit (M ,Y ,sample_weight =w )
        # 3) FOCAL -- weights KAT-DISI olasiliktan.
        # ILK DENEMEM BOZUKTU: agirliklari baseline modelin KENDI EGITIM
        # verisindeki olasiligindan hesaplamistim. Model that ornekleri
        # ezberledigi for p~1 cikiyor, (1-p)^gamma sifira iniyor and geriye
        # only noise kaliyor -- measured: UPUN'da 0.6341 -> 0.0127.
        # Dogrusu: 2 katli ic split with KAT-DISI olasilik.
        def oof_p (Mx ,Yx ):
            pr =np .zeros (len (Yx ))
            yari =np .random .default_rng (0 ).permutation (len (Yx ))
            for h in (yari [:len (yari )//2 ],yari [len (yari )//2 :]):
                digeri =np .setdiff1d (np .arange (len (Yx )),h )
                mm =yap ().fit (Mx [digeri ],Yx [digeri ])
                pr [h ]=mm .predict_proba (Mx [h ])[:,1 ]
            return pr 

        p0 =oof_p (M [alt ],Y [alt ])
        pt =np .where (Y [alt ]==1 ,p0 ,1.0 -p0 )
        wf =np .power (np .clip (1.0 -pt ,1e-6 ,1.0 ),GAMMA )
        wf =wf /max (wf .mean (),1e-9 )# olcegi koru
        modeller ["focal"]=yap ().fit (M [alt ],Y [alt ],sample_weight =wf )
        # 4) AGIRLIK + FOCAL (same fold-disi olasilik, lower-sample on)
        wa =np .where (Y [alt ]==1 ,len (neg )/max (len (poz ),1 ),1.0 )*wf 
        wa =wa /max (wa .mean (),1e-9 )
        modeller ["agirlik_focal"]=yap ().fit (M [alt ],Y [alt ],sample_weight =wa )
        del M 

        kat_sonuc [b ]={}
        for ad ,m in modeller .items ():
            s_ic =[m .predict_proba (data_ [i ]["_M"])[:,1 ]for i in ic ]
            s_dis =[m .predict_proba (data_ [i ]["_M"])[:,1 ]for i in dis ]
            ar =np .random .default_rng (0 ).choice (
            len (ic ),min (120 ,len (ic )),replace =False )
            AR =[data_ [ic [j ]]for j in ar ]
            AS =[s_ic [j ]for j in ar ]
            rule_ =max (KURALLAR ,key =lambda k :puanla (AR ,AS ,k )["makro"])
            r =puanla ([data_ [i ]for i in dis ],s_dis ,rule_ )
            for k_ in ("TP","FP","FN"):
                top [ad ][k_ ]+=r [k_ ]
            kat_sonuc [b ][ad ]=r ["robot"]
        print (f"  {b :<6} "+" | ".join (
        f"{a } {kat_sonuc [b ][a ]:.4f}"for a in KOLLAR )
        +f"  ({time .time ()-t0 :.0f} s)",flush =True )

    last_ ={}
    for ad in KOLLAR :
        c =top [ad ]
        last_ [ad ]=2 *c ["TP"]/max (2 *c ["TP"]+c ["FP"]+c ["FN"],1 )
    print (f"\n=== KAYIP DENEMESI (MIKRO) ===")
    for ad in KOLLAR :
        fark =last_ [ad ]-last_ ["baseline"]
        print (f"  {ad :<15}{last_ [ad ]:.4f}   {fark :+.4f}"
        +("  <- KAPI GECTI"if fark >=0.01 else ""))
    json .dump ({"damga":makbuz_hash .damga (),"cluster":KUME ,"gamma":GAMMA ,
    "toplam":last_ ,"fold":kat_sonuc ,
    "not":"Kayip fonksiyonu kollari: alt-ornekleme / sinif "
    "agirligi / focal / ikisi. AYNI oznitelik, fold, kural "
    "aramasi. D7'ye BAKILMADI."},
    open (f"results/kayip_denemesi_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/kayip_denemesi_{KUME }.json")


if __name__ =="__main__":
    main ()
