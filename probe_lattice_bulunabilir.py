# -*- coding: utf-8 -*-
"""VII.0b -- ULASIM ACIGI: lattice GT olmadan, ADAYLARDAN bulunabiliyor mu?

VII.0 sunu showed: GERCEK CP'ler verildiginde 2-3 lattice onlarin %98'ini
tanimliyor (NIT). Ama that a KAHIN tavaniydi. Urun kafesi **GT'yi bilmeden**,
gurultulu candidate bulutundan bulmak zorunda.

BU SONDA that acigi olcer. Ayni acgozlu coklu-lattice aramasi, input as
GT instead of **model skorunun most high N adayini** takes. Sonra bulunan izgaranin
GERCEK GT'lerin ne kadarini yakaladigi olculur.

    kapsama(GT'den)   = yapisal ceiling        (VII.0'da measured)
    kapsama(adaydan)  = ULASILABILIR ceiling   (this probe)
    aradaki difference      = ULASIM ACIGI

Bu number, 0.70'in real olasiligini belirleyen single sayidir. Yuksek cikarsa
spread kolu kurulur; low cikarsa before candidate kalitesi duzeltilmelidir.

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
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402

KUME =os .environ .get ("KB_KUME","d6")
KAT_MIN =int (os .environ .get ("KB_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
TOL =float (os .environ .get ("KB_TOL","2.0"))# urun metrigiyle same lateral kutu
UST_N =int (os .environ .get ("KB_UST_N","150"))# kac candidate seed havuzuna girer
N_KAFES =int (os .environ .get ("KB_N","3"))


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def _izgara (P ,seed ,step_ ,n_max =60 ):
    """seed + n*step noktalari (each two yonde), part sinirlari inside."""
    L =float (np .linalg .norm (step_ ))
    if L <0.5 :
        return np .zeros ((0 ,3 ))
    u =step_ /L 
    t =(P -seed )@u 
    n0 ,n1 =int (np .floor (t .min ()/L ))-1 ,int (np .ceil (t .max ()/L ))+1 
    n0 ,n1 =max (n0 ,-n_max ),min (n1 ,n_max )
    ns =np .arange (n0 ,n1 +1 )
    return seed [None ,:]+ns [:,None ]*(u *L )[None ,:]


def _kapsa (uretilen ,G ):
    if not len (uretilen )or not len (G ):
        return np .zeros (len (G ),bool )
    d =np .linalg .norm (G [:,None ,:]-uretilen [None ,:,:],axis =-1 )
    return d .min (1 )<=TOL 


def kafes_bul (P ,G ,n_kafes =N_KAFES ):
    """ADAY bulutundan acgozlu lattice; each turda GT kapsamasi olculur."""
    kalan =np .ones (len (G ),bool )
    kaps =[]
    if len (P )<3 :
        return [0.0 ]*n_kafes 
    farklar =(P [:,None ,:]-P [None ,:,:]).reshape (-1 ,3 )
    boy =np .linalg .norm (farklar ,axis =1 )
    iyi =(boy >1.0 )&(boy <np .percentile (boy [boy >1.0 ],50 ))
    candidate =farklar [iyi ]
    if not len (candidate ):
        return [0.0 ]*n_kafes 
    rng =np .random .default_rng (0 )
    if len (candidate )>250 :
        candidate =candidate [rng .choice (len (candidate ),250 ,replace =False )]
    candidate =np .vstack ([candidate ,candidate /2.0 ,candidate /3.0 ])# lower harmonikler
    tohumlar =P [::max (1 ,len (P )//10 )]
    for _ in range (n_kafes ):
        en_m ,en_n =None ,0 
        for step_ in candidate :
            for seed in tohumlar :
                uret =_izgara (P ,seed ,step_ )
                m =_kapsa (uret ,G )&kalan 
                n =int (m .sum ())
                if n >en_n :
                    en_m ,en_n =m ,n 
        if en_m is None or en_n <1 :
            break 
        kalan =kalan &~en_m 
        kaps .append (1.0 -kalan .mean ())
    while len (kaps )<n_kafes :
        kaps .append (kaps [-1 ]if kaps else 0.0 )
    return kaps 


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
        poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        m =HistGradientBoostingClassifier (
        max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
        del M 
        for i in dis :
            oof [i ]=m .predict_proba (data_ [i ]["_M"])[:,1 ]
        print (f"  OOF {b } ({time .time ()-t0 :.0f} s)",flush =True )

    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for d ,s in zip (data_ ,oof ):
        if s is None :
            continue 
        G =np .asarray (d ["G"],float )
        if len (G )<3 :
            continue 
            # EN YUKSEK SKORLU adaylarin KONUMLARI (secenek -> candidate)
        P =d ["P"][d ["idx"]]
        rank_ =np .argsort (-np .asarray (s ))[:UST_N ]
        Pu =np .unique (np .round (P [rank_ ],3 ),axis =0 )
        kaps =kafes_bul (Pu ,G )
        a =ist [d ["mfg"]]
        a ["gt"].append (len (G ))
        for i ,k in enumerate (kaps ):
            a [f"k{i +1 }"].append (k *len (G ))
        n +=1 
        if n %40 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

            # VII.0'daki KAHIN tavani (karsilastirma for)
    oracle_ ={"NIT":0.983 ,"SUPU":0.864 ,"MOR":0.811 ,"UPUN":0.810 }
    print (f"\n{'brand':<7}{'GT':>7}{'1 lattice':>9}{'2 lattice':>9}{'3 lattice':>9}"
    f"{'KAHIN':>8}{'ULASIM ACIGI':>14}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =sum (a ["gt"])
        r ={"gt":g }
        for i in (1 ,2 ,3 ):
            r [f"lattice{i }"]=sum (a [f"k{i }"])/max (g ,1 )
        kh =oracle_ .get (m_ ,0.0 )
        r ["kahin"]=kh 
        r ["ulasim_acigi"]=kh -r ["kafes3"]
        out [m_ ]=r 
        print (f"{m_ :<7}{g :>7}{r ['kafes1']:>9.3f}{r ['kafes2']:>9.3f}"
        f"{r ['kafes3']:>9.3f}{kh :>8.3f}{r ['ulasim_acigi']:>14.3f}")
    json .dump ({"tol":TOL ,"ust_n":UST_N ,"brand":out ,
    "not":"Kafes GT'den DEGIL, model skorunun en yuksek N adayindan "
    "araniyor. ULASIM ACIGI = kahin tavani - adaydan bulunan. "
    "D7'ye BAKILMADI."},
    open (f"results/kafes_bulunabilir_{KUME }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/kafes_bulunabilir_{KUME }.json")
    print ("OKUMA: aciak KUCUKSE yayilim kolu kurulabilir; BUYUKSE first candidate "
    "kalitesi/siralamasi duzeltilmeli.")


if __name__ =="__main__":
    main ()
