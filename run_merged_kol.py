# -*- coding: utf-8 -*-
"""BIRLESIK KOL: gecen two kaldiraci BIRLIKTE kos and uctan uca olc

Bugun 19 arm measured, ikisi whereas yaradi:
  kanonik blogu           +0.0151  (parcayi own PCA cercevesine oturtur)
  spread, regime kapili   +0.0394  (coken markada kafesle uretim, d6)

Ikisi FARKLI hatalari duzeltiyor: kanonik brand bagimsizligini, spread
coken markadaki TEKRARLARI. Bu betik ikisini BIRLIKTE runs.

DORT KOL:
  baseline            : bugunku (temel features, threshold kurali)
  kanonik          : + kanonik blogu
  spread          : baseline + coken parcada lattice uretimi
  birlesik         : ikisi birden

REJIM: spread only "coken" parcada devreye girer. Olcut, parcanin skor
ayrimi (S7'nin poz-neg olcusunun urun surumu): ayrim dusukse model that parcada
sirala(ya)miyor demektir.

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
import canonical_alignment as KH # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402
from probe_lattice_v2 import kafes_ara # noqa: E402

KUME =os .environ .get ("BK_KUME","d6")
KAT_MIN =int (os .environ .get ("BK_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
MESH ={"d6":"results/_p1_olasilik",
"tam":"results/_p1_olasilik_brepegit"}[KUME ]
NMS =5.0 
KURAL =("goreli",0.85 ,0.20 )
AYRIM_ESIK =float (os .environ .get ("BK_AYRIM","0.30"))
YAKIN_R =2.0 


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def kanonik_blok (d ):
    mf =f"{MESH }/{d ['pid']}.npz"
    V =(np .asarray (np .load (mf )["V"],float )
    if os .path .exists (mf )else np .asarray (d ["P"],float ))
    return KH .oznitelik (d ["P"][d ["idx"]],d ["YD"],V ).astype (np .float32 )


def yayilim_sec (d ,s ,k_hedef ):
    """Kafesle uret, yakin seceneklerden KIPSEL yonu ver, skorla first k."""
    P =np .asarray (d ["P"],float )
    idx =np .asarray (d ["idx"],int )
    YD =_birim (np .asarray (d ["YD"],float ))
    Pu =np .unique (np .round (P ,3 ),axis =0 )
    bul =kafes_ara (Pu )
    if not bul :
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    uret =np .vstack ([b [2 ]for b in bul ])
    ust =np .argsort (-s )[:max (30 ,k_hedef )]
    Y0 =YD [ust ]
    cos =np .clip (Y0 @Y0 .T ,-1 ,1 )
    oy =(np .degrees (np .arccos (cos ))<=K .ACI ).sum (1 )
    kipsel =Y0 [int (np .argmax (oy ))]
    d_ua =np .linalg .norm (uret [:,None ,:]-P [None ,:,:],axis =-1 )
    skor =np .full (len (uret ),-1.0 )
    for u in range (len (uret )):
        ad =np .where (d_ua [u ]<=YAKIN_R )[0 ]
        if not len (ad ):
            continue 
        m_ =np .isin (idx ,ad )
        if m_ .any ():
            skor [u ]=s [m_ ].max ()
    g =skor >=0 
    if not g .any ():
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    U ,S_ =uret [g ],skor [g ]
    sp =[]
    for j in np .argsort (-S_ ):
        if len (sp )>=k_hedef :
            break 
        p =U [j ]
        if sp and min (np .linalg .norm (np .asarray (sp )-p ,axis =1 ))<NMS :
            continue 
        sp .append (p )
    SP =np .asarray (sp ).reshape (-1 ,3 )
    return SP ,np .tile (kipsel ,(len (SP ),1 ))


def f1 (t ,f ,n ):
    return 2 *t /max (2 *t +f +n ,1 )


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
        d ["_K"]=kanonik_blok (d )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar } | ayrim esigi {AYRIM_ESIK }",
    flush =True )

    skor ={"baseline":[None ]*len (data_ ),"kanonik":[None ]*len (data_ )}
    for b in katlar :
        ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
        for ad in ("baseline","kanonik"):
            def mat (i ):
                return (data_ [i ]["_M"]if ad =="baseline"
                else np .hstack ([data_ [i ]["_M"],data_ [i ]["_K"]]))
            n_s =sum (len (data_ [i ]["y"])for i in ic )
            M =np .empty ((n_s ,mat (ic [0 ]).shape [1 ]),np .float32 )
            o =0 
            for i in ic :
                m_ =mat (i )
                M [o :o +len (m_ )]=m_ 
                o +=len (m_ )
            Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
            rng =np .random .default_rng (0 )
            poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
            sec =np .concatenate ([poz ,rng .choice (
            neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
            m =HistGradientBoostingClassifier (
            max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
            l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
            del M 
            for i in dis :
                skor [ad ][i ]=m .predict_proba (mat (i ))[:,1 ]
        print (f"  {b } bitti ({time .time ()-t0 :.0f} s)",flush =True )

    KOLLAR =("baseline","kanonik","yayilim","birlesik")
    agg =collections .defaultdict (lambda :collections .Counter ())
    for d in data_ :
        if skor ["baseline"][data_ .index (d )]is None :
            continue 
        i =data_ .index (d )
        G =np .asarray (d ["G"],float )
        Gd =np .asarray (d ["Gd"],float )
        dg =d ["diag"]
        a =agg [d ["mfg"]]
        a ["gt"]+=len (G )
        for arm in KOLLAR :
            s =skor ["kanonik"if arm in ("kanonik","birlesik")
            else "baseline"][i ]
            ayrim =float (np .percentile (s ,99 )-np .median (s ))
            kafesli =arm in ("yayilim","birlesik")and ayrim <AYRIM_ESIK 
            if kafesli :
                P ,D =yayilim_sec (d ,s ,max (len (G ),4 ))
            else :
                P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,KURAL ,
                nms_mm =NMS )
            tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
            signed =True )[:3 ]
            a [f"{arm }_tp"]+=tp 
            a [f"{arm }_fp"]+=fp 
            a [f"{arm }_fn"]+=fn 

    print (f"\n{'brand':<7}{'GT':>7}"+"".join (f"{k :>11}"for k in KOLLAR ))
    T =collections .Counter ()
    out ={}
    for m_ in sorted (agg ,key =lambda x :-agg [x ]["gt"]):
        a =agg [m_ ]
        for k_ in a :
            T [k_ ]+=a [k_ ]
        r ={arm :f1 (a [f"{arm }_tp"],a [f"{arm }_fp"],a [f"{arm }_fn"])
        for arm in KOLLAR }
        r ["gt"]=a ["gt"]
        out [m_ ]=r 
        print (f"{m_ :<7}{a ['gt']:>7}"+
        "".join (f"{r [k ]:>11.4f}"for k in KOLLAR ))
    last_ ={arm :f1 (T [f"{arm }_tp"],T [f"{arm }_fp"],T [f"{arm }_fn"])
    for arm in KOLLAR }
    print (f"{'TOPLAM':<7}{T ['gt']:>7}"+
    "".join (f"{last_ [k ]:>11.4f}"for k in KOLLAR ))
    print ("\n=== TABANA GORE ===")
    for arm in KOLLAR [1 :]:
        fark =last_ [arm ]-last_ ["baseline"]
        print (f"  {arm :<12}{last_ [arm ]:.4f}   {fark :+.4f}"
        +("  <- KAZANC"if fark >0 else ""))
    json .dump ({"damga":makbuz_hash .damga (),"cluster":KUME ,
    "ayrim_esik":AYRIM_ESIK ,"toplam":last_ ,"brand":out ,
    "not":"Gecen iki kaldirac BIRLIKTE: kanonik blogu + regime "
    "kapili lattice yayilimi. D7'ye BAKILMADI."},
    open (f"results/birlesik_kol_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/birlesik_kol_{KUME }.json")


if __name__ =="__main__":
    main ()
