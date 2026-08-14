# -*- coding: utf-8 -*-
"""VII.0l/VII.1 -- SUZGEC: 179 uretilen noktayi ~k'ya indirirken dogrulari tut

VII.4 kolun CANLI oldugunu showed (NIT'te full kabul kutusuyla kapsama
0.527, bugunku F1 0.0089). AMA arm ham haliyle TABANDAN KOTU (0.0845 vs
0.1789) because part basina 179 point yayinlamak kesinligi olduruyor.

Kolun butun degeri SUZGECTE. Bu probe most ucuz suzgeci olcer: uretilen each
izgara noktasini MEVCUT MODEL SKORUYLA puanla (cevresindeki seceneklerin most
high skoru), sirala, first k'yi al.

OLCU: first k'da remaining kapsama (full kabul kutusu) and elde edilen F1.
Kiyas noktasi: suzgecsiz kapsama (VII.4) and bugunku F1.

Suzgec CALISIRSA arm kurulabilir; calismazsa isin/mesh dogrulamasi (hole
gercekten present mi) and kipsel imza denenmeli.

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

KUME =os .environ .get ("KS_KUME","d6")
KAT_MIN =int (os .environ .get ("KS_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
YANAL ,EKSENEL =2.0 ,40.0 
YAKIN_R =2.0 
NMS =5.0 


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar }",flush =True )

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
        Gd =np .asarray (d ["Gd"],float )
        Gn =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
        P =np .asarray (d ["P"],float )
        idx =np .asarray (d ["idx"],int )
        YD =np .asarray (d ["YD"],float )
        s =np .asarray (s ,float )
        Pu =np .unique (np .round (P ,3 ),axis =0 )
        bul =kafes_ara (Pu )
        if not bul :
            continue 
        uret =np .vstack ([b [2 ]for b in bul ])

        # each uretilen point for: yakin seceneklerin EN IYI skoru + that direction
        d_ua =np .linalg .norm (uret [:,None ,:]-P [None ,:,:],axis =-1 )
        score =np .full (len (uret ),-1.0 )
        direction =np .zeros ((len (uret ),3 ))
        for u in range (len (uret )):
            ad =np .where (d_ua [u ]<=YAKIN_R )[0 ]
            if not len (ad ):
                continue 
            m_ =np .isin (idx ,ad )
            if not m_ .any ():
                continue 
            j =np .argmax (np .where (m_ ,s ,-1 ))
            score [u ]=s [j ]
            direction [u ]=YD [j ]
            # KIPSEL YON (part own sablonunu tanimlar): a klemenste butun
            # CP'ler PARALELDIR. Yonu point basina secmek zayif modele guvenmek
            # demek -- measured: correct direction MEVCUT (oracle 0.527) but most high
            # skorlu secenegin yonu most zaman YANLIS (0.054).
            # K2.1 yonu TEK TOHUMDAN kopyalayip cokmustu; KIPSEL direction, high
            # skorlu seceneklerin on oy birligiyle belirlenir.
        if os .environ .get ("KS_KIPSEL","1")=="1":
            ust =np .argsort (-s )[:max (30 ,len (G ))]
            Y0 =YD [ust ]
            Y0 =Y0 /np .maximum (np .linalg .norm (Y0 ,axis =1 ,keepdims =True ),
            1e-12 )
            # ISARETLI oy: each direction adayina kac option 10 derece inside
            cos =np .clip (Y0 @Y0 .T ,-1 ,1 )
            oy =(np .degrees (np .arccos (cos ))<=K .ACI ).sum (1 )
            kipsel =Y0 [int (np .argmax (oy ))]
            direction [:]=kipsel 
        valid =score >=0 
        if not valid .any ():
            continue 
        U ,S_ ,D_ =uret [valid ],score [valid ],direction [valid ]
        rank_ =np .argsort (-S_ )

        k =len (G )
        a =ist [d ["mfg"]]
        a ["gt"].append (k )
        a ["uret"].append (len (U ))
        for label_ ,kk in (("hepsi",len (U )),("k",k ),("2k",2 *k )):
            secP ,secD =[],[]
            for j in rank_ :
                if len (secP )>=kk :
                    break 
                p =U [j ]
                if secP and min (np .linalg .norm (np .asarray (secP )-p ,
                axis =1 ))<NMS :
                    continue 
                secP .append (p )
                secD .append (D_ [j ])
            if not secP :
                a [f"tp_{label_ }"].append (0 )
                a [f"n_{label_ }"].append (0 )
                continue 
            SP =np .asarray (secP )
            SD =np .asarray (secD )
            SD =SD /np .maximum (np .linalg .norm (SD ,axis =1 ,keepdims =True ),
            1e-12 )
            v =SP [:,None ,:]-G [None ,:,:]
            al =(v *Gn [None ,:,:]).sum (-1 )
            yan =np .linalg .norm (v -al [...,None ]*Gn [None ,:,:],axis =-1 )
            aci =np .degrees (np .arccos (np .clip (SD @Gn .T ,-1 ,1 )))
            ok =(yan <=YANAL )&(np .abs (al )<=EKSENEL )&(aci <=K .ACI )
            a [f"tp_{label_ }"].append (int (ok .any (0 ).sum ()))
            a [f"n_{label_ }"].append (len (SP ))
        n +=1 
        if n %30 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    def f1 (tp ,np_ ,gt ):
        return 2 *tp /max (2 *tp +(np_ -tp )+(gt -tp ),1 )

    print (f"\n{'brand':<7}{'GT':>7}{'uret/p':>8}"+
    "".join (f"{e :>18}"for e in ("suzgecsiz","ilk k","ilk 2k")))
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =sum (a ["gt"])
        sat =f"{m_ :<7}{g :>7}{np .mean (a ['uret']):>8.0f}"
        r ={"gt":g }
        for e in ("hepsi","k","2k"):
            tp =sum (a [f"tp_{e }"])
            npr =sum (a [f"n_{e }"])
            r [e ]={"tp":tp ,"n":npr ,"recall":tp /max (g ,1 ),
            "f1":f1 (tp ,npr ,g )}
            sat +=f"{r [e ]['recall']:>9.3f}{r [e ]['f1']:>9.3f}"
        out [m_ ]=r 
        print (sat )
    print (f"\n{'':<22}"+"".join (f"{'recall':>9}{'F1':>9}"for _ in range (3 )))
    json .dump ({"brand":out ,"yakin_r":YAKIN_R ,"nms":NMS ,
    "not":"Uretilen izgara noktalari MEVCUT MODEL SKORUYLA "
    "siralanip ilk k / 2k alinir. Tam kabul kutusu. "
    "D7'ye BAKILMADI."},
    open (f"results/kafes_suzgec_{KUME }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/kafes_suzgec_{KUME }.json")


if __name__ =="__main__":
    main ()
