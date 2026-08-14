# -*- coding: utf-8 -*-
"""S5: FIZIKSEL BAYRAKLARI FIX TETIGI DEGIL, GATE OZNITELIGI OLARAK KULLAN.

S3/S4 SONUCU: onarim fiziksel kusuru yariya indiriyor (%18.7 -> %10.2) AMA metrik
kayitsiz (tespit +0.0016, robot -0.0053) and four triyajin dordu de kill'i gecemedi.
REASON MEASURED: manufacturer CP'lerinin ~%47'si GOVDENIN ICINDE; noktayi agza tasimak
fiziksel gecerlilik kazandirirken GT'den UZAKLASTIRIYOR.

S0 HARITASI BASKA BIR KULLANIM ONERIYOR:
    bayrak absent  -> FP orani %21.8
    1 bayrak    -> %49.1
    2 bayrak    -> %43.8
    3 bayrak    -> %44.4
Herhangi a bayrak FP olasiligini IKI KATINA cikariyor (count artinca artmiyor:
dereceli not IKILI sinyal). Bayraklari NOKTAYI OYNATMAK for not, gate'e
"this candidate fiziksel as suphelidir" demek for kullanmak that celiskiye GIRMEZ.

Bu betik grup-capraz a FIZIBILITE olcumudur (dagitim DEGIL): measurement kumesinin own
satirlariyla, gate'e 3 bayrak eklemenin tespit F1'ine etkisi. Gecerse training korpusunda
bayrak uretilip (approximately 1700 part, ~40 dk mesh isi) real gate yeniden egitilir.

TEZ-NOTR: bayraklar candidate duzeyinde FIZIKSEL olculerdir; network, remesh and v_o does not change.
KILL: tespit >= +0.010 VE manufacturer-disi dusmuyor.
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import measure_set 
import wire_gate 
from sina_cluster import match_greedy ,f1w 


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .metrics import roc_auc_score 
    import protocol 
    protocol .tez_dogrula ()
    DER ,gate ,ek =T2 .yukle ()
    with open ("results/_a4_bayraklar.pkl","rb")as f :
        BAY =pickle .load (f )

        # --- satirlari kur: gate ozellikleri + 3 fiziksel bayrak + label (GT'ye yakin mi)
    X0 ,X1 ,Y ,GG ,KEY =[],[],[],[],[]
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        X =np .hstack ([r ["X"],r ["XR"]])
        if X .shape [1 ]*2 !=gate ["n_feat"]:
            continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X ))
        bl =BAY .get (r ["pid"])or []
        if not k .any ()or len (bl )!=int (k .sum ()):
            continue 
        P =np .asarray (r ["P"],float )[k ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        Xk =X [k ]
        # label: this candidate GT'ye TESPIT toleransinda mi
        if len (G ):
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            tt =max (3.0 ,0.06 *float (r ["diag"]))
            iyi =(pe .min (1 )<=tt )
        else :
            iyi =np .zeros (len (P ),bool )
        for i in range (len (P )):
            b =bl [i ]
            ic =1.0 if b ["govde_ici"]else 0.0 
            il =float (b ["ileri"])if np .isfinite (b ["ileri"])else 60.0 
            cp =float (b ["ic_cap"])if np .isfinite (b ["ic_cap"])else -1.0 
            X0 .append (Xk [i ])
            X1 .append (np .concatenate ([Xk [i ],[ic ,min (il ,60.0 ),cp ,
            1.0 if il <5.0 else 0.0 ,
            1.0 if 0 <cp <0.8 else 0.0 ]]))
            Y .append (int (iyi [i ]));GG .append (r ["geo"]);KEY .append ((r ["pid"],i ))
    X0 =np .array (X0 ,float );X1 =np .array (X1 ,float )
    Y =np .array (Y );GG =np .array (GG )
    print (f"satir {len (Y )} | iyi candidate %{100 *Y .mean ():.1f} | sutun {X0 .shape [1 ]} -> {X1 .shape [1 ]}")

    ug =np .unique (GG );rng =np .random .RandomState (0 );rng .shuffle (ug )
    fold ={g :j %5 for j ,g in enumerate (ug )}
    kk =np .array ([fold [g ]for g in GG ])
    OOF ={}
    for ad ,M in (("TABAN",X0 ),("+BAYRAK",X1 )):
        o =np .zeros (len (Y ))
        for f_ in range (5 ):
            tr =kk !=f_ 
            Z =np .zeros ((len (M ),M .shape [1 ]*2 ))
            for pid in {k [0 ]for k in KEY }:
                idx =[j for j ,k in enumerate (KEY )if k [0 ]==pid ]
                Z [idx ]=wire_gate .within_part (M [idx ],"zskor")
            c =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (Z [tr ],Y [tr ])
            o [~tr ]=c .predict_proba (Z [~tr ])[:,1 ]
        OOF [ad ]=o 
        print (f"  {ad :<10} grup-capraz AUC {roc_auc_score (Y ,o ):.4f}")

        # --- uctan uca: OOF skoruyla karar kurali
    SC ={ad :{}for ad in OOF }
    for ad ,o in OOF .items ():
        for j ,key in enumerate (KEY ):
            SC [ad ].setdefault (key [0 ],{})[key [1 ]]=o [j ]

    def kos (ad ):
        rows =[]
        for r in DER :
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="very"if r ["n"]>=8 else "low"
            s =SC [ad ].get (r ["pid"])
            if s is None or r ["X"]is None :
                rows .append ((rj ,0 ,0 ,len (G )));continue 
            X =np .hstack ([r ["X"],r ["XR"]])
            k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X ))
            P =np .asarray (r ["P"],float )[k ];D =np .asarray (r ["Pd"],float )[k ]
            sk =np .array ([s .get (i ,0.0 )for i in range (len (P ))])
            m =wire_gate .decision_mask (sk )
            tp ,fp ,fn ,_ =match_greedy (P [m ],D [m ],G ,Gd ,r ["diag"],0.0 ,180.0 ,True )
            rows .append ((rj ,tp ,fp ,fn ))
        return f1w (rows )

    a ,b =kos ("TABAN"),kos ("+BAYRAK")
    print (f"\nuctan uca tespit: TABAN {a :.4f} -> +BAYRAK {b :.4f}  ({b -a :+.4f})")
    gecti =(b -a )>=0.010 
    print (f"KILL: tespit >= +0.010 -> {'GECTI -- training korpusunda bayrak uretilir'if gecti else 'GECMEDI'}")
    with io .open ("results/s5_bayrak_ozellik.json","w",encoding ="utf-8")as f :
        json .dump ({"auc":{k :float (roc_auc_score (Y ,v ))for k ,v in OOF .items ()},
        "tespit_taban":a ,"tespit_bayrak":b ,"difference":b -a ,
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/s5_bayrak_ozellik.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
