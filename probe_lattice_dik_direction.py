# -*- coding: utf-8 -*-
"""KAFES ADIMI YONU KISITLAR: axis, siraya DIKTIR

FIKIR (denenmemis). Bir order deligin ekseni, siranin uzandigi yone DIKTIR.
Kafes aramasi already step vektorunu (siranin yonunu) veriyor. O halde:

    correct direction, step vektorune DIK which is duzlemde yatar

Bu, yonu 3 boyutlu a secim olmaktan cikarip 1 boyutlu a cembere indirir.
Ustune "govdeden disari" and "part ici paralellik" eklenince neredeyse
tekleser.

WHY IMPORTANT. Yayilim kolunu 0.527'den 0.077'ye dusuren sey YON SECIMIYDI.
Bu kisit never kullanilmadi.

OLCULEN (uretilen konumlarda, TAM kabul kutusu):
  all of them       : yakin seceneklerin most high skorlusu (bugunku, 0.077)
  dik_suzgec  : before adima DIK olanlari suz, after most high skorlu
  dik_kipsel  : dik olanlar inside part-ici KIPSEL direction
  kahin       : correct direction MEVCUT mu (upper boundary, 0.527)

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
from probe_lattice_v2 import kafes_ara # noqa: E402

KUME =os .environ .get ("KY_KUME","d6")
KAT_MIN =int (os .environ .get ("KY_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
YANAL ,EKSENEL =2.0 ,40.0 
YAKIN_R =2.0 
DIK_TOL =float (os .environ .get ("KY_DIK","15.0"))# 90 dereceden deviation


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar } | dik tol {DIK_TOL }",
    flush =True )

    oof =[None ]*len (data_ )
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
        Gn =_birim (np .asarray (d ["Gd"],float ))
        P =np .asarray (d ["P"],float )
        idx =np .asarray (d ["idx"],int )
        YD =_birim (np .asarray (d ["YD"],float ))
        s =np .asarray (s ,float )
        Pu =np .unique (np .round (P ,3 ),axis =0 )
        bul =kafes_ara (Pu )
        if not bul :
            continue 

        say ={a :0 for a in ("hepsi","dik_suzgec","dik_kipsel","kahin")}
        for (seed ,step_ ),_puan ,uret in bul :
            u =_birim (step_ .reshape (1 ,3 ))[0 ]
            v =uret [:,None ,:]-G [None ,:,:]
            al =(v *Gn [None ,:,:]).sum (-1 )
            yan =np .linalg .norm (v -al [...,None ]*Gn [None ,:,:],axis =-1 )
            konum =(yan <=YANAL )&(np .abs (al )<=EKSENEL )
            if not konum .any ():
                continue 
            d_ua =np .linalg .norm (uret [:,None ,:]-P [None ,:,:],axis =-1 )
            for j in range (len (G )):
                u_ler =np .where (konum [:,j ])[0 ]
                if not len (u_ler ):
                    continue 
                ad =np .where ((d_ua [u_ler ]<=YAKIN_R ).any (0 ))[0 ]
                if not len (ad ):
                    continue 
                m_ =np .isin (idx ,ad )
                if not m_ .any ():
                    continue 
                Yo ,So =YD [m_ ],s [m_ ]
                aci_gt =np .degrees (np .arccos (np .clip (Yo @Gn [j ],-1 ,1 )))
                # KAHIN: correct direction mevcut mu
                if (aci_gt <=K .ACI ).any ():
                    say ["kahin"]+=1 
                    # BUGUNKU: most high skorlu
                if aci_gt [int (np .argmax (So ))]<=K .ACI :
                    say ["hepsi"]+=1 
                    # DIK SUZGEC: adima dik olanlar
                deviation =np .abs (90.0 -np .degrees (
                np .arccos (np .clip (np .abs (Yo @u ),-1 ,1 ))))
                dik =deviation <=DIK_TOL 
                if dik .any ():
                    Yd ,Sd ,Ad =Yo [dik ],So [dik ],aci_gt [dik ]
                    if Ad [int (np .argmax (Sd ))]<=K .ACI :
                        say ["dik_suzgec"]+=1 
                        # KIPSEL: dik olanlar inside at most oy alan direction
                    cos =np .clip (Yd @Yd .T ,-1 ,1 )
                    oy =(np .degrees (np .arccos (cos ))<=K .ACI ).sum (1 )
                    if Ad [int (np .argmax (oy ))]<=K .ACI :
                        say ["dik_kipsel"]+=1 
        a =ist [d ["mfg"]]
        a ["gt"].append (len (G ))
        for k_ ,v_ in say .items ():
            a [k_ ].append (min (v_ ,len (G )))
        n +=1 
        if n %40 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    print (f"\n{'brand':<7}{'GT':>7}{'BUGUNKU':>10}{'DIK suzgec':>12}"
    f"{'DIK kipsel':>12}{'KAHIN':>9}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =max (sum (a ["gt"]),1 )
        r ={k :sum (a [k ])/g for k in ("hepsi","dik_suzgec","dik_kipsel",
        "kahin")}
        r ["gt"]=g 
        out [m_ ]=r 
        print (f"{m_ :<7}{g :>7}{r ['hepsi']:>10.3f}{r ['dik_suzgec']:>12.3f}"
        f"{r ['dik_kipsel']:>12.3f}{r ['kahin']:>9.3f}")
    json .dump ({"dik_tol":DIK_TOL ,"brand":out ,
    "not":"Kafes adimi yonu kisitlar: axis siraya DIKTIR. "
    "Uretilen konumlarda direction secimi. D7'ye BAKILMADI."},
    open (f"results/kafes_dik_yon_{KUME }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/kafes_dik_yon_{KUME }.json")
    print ("OKUMA: DIK suzgec BUGUNKUyu asiyorsa kisit whereas yariyor;")
    print ("       KAHIN'e yaklasiyorsa direction secimi COZULMUS demektir.")


if __name__ =="__main__":
    main ()
