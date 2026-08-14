# -*- coding: utf-8 -*-
"""TEKRAR SONDASI: model ILK CP'yi buluyor da TEKRARLARI mi bulamiyor?

TESHISIMDEKI DEFECT (2026-08-12). S7'de "correct secenegin order yuzdeligi 0.005"
diye olctugum sey, parcadaki **EN IYI siralanmis** correct secenegin sirasiydi.
Yani "a tane correct secenek tepeye yakin" diyor -- 24 CP'li a parcada
24.'sunun nerede oldugu hakkinda HICBIR SEY soylemiyor. "Siralama iyi"
cikarimim that is why extra iyimserdi.

DOGRU SORU: k'inci correct secenek kacinci sirada? (k = parcanin CP count)

HIPOTEZ: model ILK CP'yi buluyor, TEKRARLARI bulamiyor. Bu, gozlenen deseni
birebir aciklar -- few CP'li brand (UPUN 3.2) calisiyor, very CP'li (NIT 24.4)
cokuyor; because bugun HER CP own basina markalar-arasi a karar gerektiriyor.

AYRICA OLCULUR -- TEKRAR TAVANI: most iyi tohumdan TEK BIR OTELEME with kac GT
uretilebilir? Bu, "kendine benzerlik with spread" kolunun upper siniridir and
markalar-arasi transfer GEREKTIRMEZ: gorulmemis markada da 24 hole
birbirinin aynisidir.

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

KUME =os .environ .get ("TK_KUME","d6")
KAT_MIN =int (os .environ .get ("TK_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
EKSEN_TOL =40.0 
OTEL_TOL =1.0 # oteleme with uretilen point GT'ye this up to yakinsa TUTAR


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])]).astype (
    np .float32 )


def dogru_maske (d ):
    P =d ["P"][d ["idx"]]
    YD =d ["YD"]
    G =np .asarray (d ["G"],float )
    Gd =np .asarray (d ["Gd"],float )
    if not len (P )or not len (G ):
        return np .zeros (len (P ),bool )
    Gd =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
    v =P [:,None ,:]-G [None ,:,:]
    al =(v *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (v -al [...,None ]*Gd [None ,:,:],axis =-1 )
    an =np .degrees (np .arccos (np .clip (YD @Gd .T ,-1 ,1 )))
    return ((pe <=K .YANAL )&(np .abs (al )<=EKSEN_TOL )&(an <=K .ACI ))


def tekrar_tavani (G ):
    """EN IYI TEK OTELEME with kac GT uretilebilir (a seed + n*step)?

    Butun GT ciftlerinin difference vektorleri candidate adimdir; each candidate for
    'seed + n*step' izgarasina OTEL_TOL inside dusen GT sayilir.
    """
    G =np .asarray (G ,float )
    n =len (G )
    if n <3 :
        return n ,None 
    en_iyi ,en_adim =1 ,None 
    # candidate adimlar: most yakin komsu farklari (all of them not -- O(n^2) yeter)
    farklar =(G [:,None ,:]-G [None ,:,:]).reshape (-1 ,3 )
    boy =np .linalg .norm (farklar ,axis =1 )
    candidate =farklar [(boy >0.5 )&(boy <np .percentile (boy [boy >0.5 ],40 ))]
    if not len (candidate ):
        return 1 ,None 
    if len (candidate )>400 :
        candidate =candidate [np .random .default_rng (0 ).choice (len (candidate ),400 ,False )]
    for adim in candidate :
        L =np .linalg .norm (adim )
        u =adim /L 
        for seed in G :
            t =(G -seed )@u 
            dik =np .linalg .norm ((G -seed )-t [:,None ]*u [None ,:],axis =1 )
            k =np .round (t /L )
            error =np .abs (t -k *L )
            tut =int (((dik <=OTEL_TOL )&(error <=OTEL_TOL )).sum ())
            if tut >en_iyi :
                en_iyi ,en_adim =tut ,float (L )
    return en_iyi ,en_adim 


def main ():
    t0 =time .time ()
    veri =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in veri :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
    brand =collections .Counter (d ["mfg"]for d in veri )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (veri )} part | katlar {katlar } ({time .time ()-t0 :.0f} s)",
    flush =True )

    oof =[None ]*len (veri )
    for b in katlar :
        ic =[i for i ,d in enumerate (veri )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (veri )if d ["mfg"]==b ]
        n_satir =sum (len (veri [i ]["y"])for i in ic )
        M =np .empty ((n_satir ,veri [0 ]["_M"].shape [1 ]),np .float32 )
        o =0 
        for i in ic :
            m_ =veri [i ]["_M"]
            M [o :o +len (m_ )]=m_ 
            o +=len (m_ )
        Y =np .concatenate ([veri [i ]["y"]for i in ic ])
        rng =np .random .default_rng (0 )
        poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        m =HistGradientBoostingClassifier (
        max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
        del M 
        for i in dis :
            oof [i ]=m .predict_proba (veri [i ]["_M"])[:,1 ]
        print (f"  OOF {b } ({time .time ()-t0 :.0f} s)",flush =True )

    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    for d ,s in zip (veri ,oof ):
        if s is None :
            continue 
        a =ist [d ["mfg"]]
        G =np .asarray (d ["G"],float )
        k =len (G )
        a ["cp"].append (k )
        dg =dogru_maske (d )# (secenek, GT)
        if not dg .any ():
            continue 
        sira =np .argsort (np .argsort (-np .asarray (s )))
        # HER GT for: onu tutan seceneklerin EN IYI order
        gt_sira =[]
        for j in range (dg .shape [1 ]):
            u =np .where (dg [:,j ])[0 ]
            if len (u ):
                gt_sira .append (int (sira [u ].min ()))
        if not gt_sira :
            continue 
        gt_sira =np .sort (gt_sira )
        a ["ilk_gt_sira"].append (int (gt_sira [0 ]))
        a ["son_gt_sira"].append (int (gt_sira [-1 ]))
        a ["ortanca_gt_sira"].append (float (np .median (gt_sira )))
        # first k sirada kac GT present (k = real CP count)
        a ["ilkk_icinde"].append (float ((gt_sira <max (k ,1 )).mean ()))
        # TEKRAR TAVANI
        tut ,adim =tekrar_tavani (G )
        a ["tekrar_orani"].append (tut /max (k ,1 ))
        if adim :
            a ["adim"].append (adim )

    print (f"\n{'brand':<7}{'CP/p':>7}{'ILK GT':>9}{'ORTANCA':>9}{'SON GT':>9}"
    f"{'ilk-k icinde':>14}{'TEKRAR tavani':>15}{'adim mm':>9}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-np .mean (ist [x ]["cp"])):
        a =ist [m_ ]
        if not a ["ilk_gt_sira"]:
            continue 
        r ={"cp_parca":float (np .mean (a ["cp"])),
        "ilk_gt_sira":float (np .median (a ["ilk_gt_sira"])),
        "ortanca_gt_sira":float (np .median (a ["ortanca_gt_sira"])),
        "son_gt_sira":float (np .median (a ["son_gt_sira"])),
        "ilkk_icinde":float (np .mean (a ["ilkk_icinde"])),
        "tekrar_tavani":float (np .mean (a ["tekrar_orani"])),
        "adim_mm":float (np .median (a ["adim"]))if a ["adim"]else 0.0 }
        out [m_ ]=r 
        print (f"{m_ :<7}{r ['cp_parca']:>7.1f}{r ['ilk_gt_sira']:>9.0f}"
        f"{r ['ortanca_gt_sira']:>9.0f}{r ['son_gt_sira']:>9.0f}"
        f"{r ['ilkk_icinde']:>14.3f}{r ['tekrar_tavani']:>15.3f}"
        f"{r ['adim_mm']:>9.2f}")
    print ("\nOKUMA:")
    print ("  ILK GT << SON GT  -> model ILKINI buluyor, TEKRARLARI bulamiyor")
    print ("  TEKRAR tavani yuksek -> tek bir otelemeyle GT'lerin o orani")
    print ("     URETILEBILIR; markalar-arasi transfer GEREKMEZ")
    json .dump ({"dizin":os .environ ["P6_DIZIN"],"cluster":KUME ,"brand":out ,
    "not":"GT sirasi = o GT'yi tutan seceneklerin EN IYI sirasi. "
    "TEKRAR tavani = en iyi TEK oteleme izgarasina dusen GT "
    "orani (kendine-benzerlik yayiliminin ust siniri). "
    "D7'ye BAKILMADI."},
    open (f"results/tekrar_sondasi_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/tekrar_sondasi_{KUME }.json")


if __name__ =="__main__":
    main ()
