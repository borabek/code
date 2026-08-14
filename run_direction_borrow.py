# -*- coding: utf-8 -*-
"""YON ODUNC ALMA kolunu egit and D7'de (brand-disi) uctan uca olc.

Kol candidate EKLEMEZ, only present which is adayin YONUNU degistirebilir -> FP count
artamaz. Bugun closed five mimarinin all of them candidate ekliyordu; this yapisal as
different.

EGITIM: D6 (468) + corpus (G7 eksi D7). Etiket, YALNIZCA konumu already robot
olcutunu saglayabilen adaylarda tanimlidir -- direction karari however orada ANLAMLIDIR;
konumu tutmayan adayi egitime katmak siniflandiriciyi cozulemez orneklerle
bogar (all of them negatif).

OLCUM: urunun own zinciri (gate v6 -> karar kurali -> NMS), single difference yonlerin
secilmesi. MIKRO, D7 brand-disi.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 
from sklearn .ensemble import RandomForestClassifier 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402
import wire_gate # noqa: E402
import direction_borrow as YO # noqa: E402
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 


def warn_position (p ,g ,gd ):
    """Bu KONUM, yonden bagimsiz as robot olcutunu saglayabilir mi?"""
    n =np .linalg .norm (gd )
    if n <1e-9 :
        return False 
    u =gd /n 
    v =p -g 
    e =float (v @u )
    return (np .linalg .norm (v -e *u )<=YANAL )and (abs (e )<=EKSENEL )


def egitim_verisi (kayitlar ,gate ,cyl ):
    X ,Y =[],[]
    for pid ,r in kayitlar .items ():
        G =np .asarray (r .get ("G",[]),float )
        M =K .x58 (r )if "XR"in r else None 
        if M is None or not len (G ):
            continue 
        P =np .asarray (r ["P"],float )
        D =np .asarray (r ["Pd"],float )
        if len (P )<1 :
            continue 
        Gd =np .asarray (r ["Gd"],float )
        s =np .asarray (wire_gate .decision_score (gate ,M ),float )
        k =maske (s ,0.40 ,0.30 )
        if not k .any ():
            continue 
        Pk ,Dk ,sk =P [k ],D [k ],s [k ]
        bask =YO .baskin_yon (Dk )
        cy =cyl .get (str (pid ))
        for i in range (len (Pk )):
        # this adayin KONUMUNU saglayan GT present mi
            j =-1 
            for t in range (len (G )):
                if warn_position (Pk [i ],G [t ],Gd [t ]):
                    j =t 
                    break 
            if j <0 :
                continue # direction karari here ANLAMSIZ
            V ,F =YO .options (Pk ,Dk ,i ,cy ,float (sk [i ]),bask )
            if len (V )<2 :
                continue 
            u =Gd [j ]/max (np .linalg .norm (Gd [j ]),1e-12 )
            for q ,v in enumerate (V ):
                a =np .degrees (np .arccos (np .clip (abs (float (v @u )),-1.0 ,1.0 )))
                X .append (F [q ]);Y .append (int (a <=ACI ))
    return np .asarray (X ,float ),np .asarray (Y ,int )


def main ():
    gate =K .gate_yukle ()
    cy6 =pickle .load (open ("results/_d6_silindirler.pkl","rb"))
    cyk =pickle .load (open ("results/_brepegit_silindirler.pkl","rb"))
    cy7 =pickle .load (open ("results/_d7_silindirler.pkl","rb"))

    d6 =d6_record .yukle (set (d6_record .exam ()["pidler"]))
    kor_p =[str (p )for p in json .load (
    open ("results/brep_training_set.json"))["pidler"]]
    kor =K .yukle (kor_p )
    print (f"training: D6 {len (d6 )} + corpus {len (kor )}",flush =True )

    X1 ,Y1 =egitim_verisi (d6 ,gate ,cy6 )
    print (f"  D6 option {len (X1 )} | pozitif {Y1 .mean ()if len (Y1 )else 0 :.4f}",
    flush =True )
    X2 ,Y2 =egitim_verisi (kor ,gate ,cyk )
    print (f"  corpus option {len (X2 )} | pozitif {Y2 .mean ()if len (Y2 )else 0 :.4f}",
    flush =True )
    X =np .vstack ([X1 ,X2 ]);Y =np .concatenate ([Y1 ,Y2 ])
    print (f"TOPLAM {X .shape } | pozitif {Y .mean ():.4f}",flush =True )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    class_weight ="balanced_subsample",
    random_state =0 ).fit (X ,Y )
    pickle .dump ({"clf":clf ,"oz_ad":YO .OZ_AD },
    open ("results/yon_odunc_model.pkl","wb"))

    te =K .yukle (json .load (open ("results/d7_exam_set.json"))["pidler"])
    S =K .step_map ()
    out ={}
    for ad ,arm in (("TEZ-SAF (direction degismez)",False ),("YON ODUNC",True )):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
        tes =[]
        for pid ,r in te .items ():
            G =np .asarray (r .get ("G",[]),float )
            M =K .x58 (r )
            if M is None or not len (G ):
                continue 
            P =np .asarray (r ["P"],float )
            D =np .asarray (r ["Pd"],float )
            Gd =np .asarray (r ["Gd"],float )
            dg =float (r ["diag"])
            s =np .asarray (wire_gate .decision_score (gate ,M ),float )
            k =maske (s ,0.40 ,0.30 )
            Pk ,Dk ,sk =(P [k ],D [k ],s [k ])if k .any ()else (P [:0 ],D [:0 ],s [:0 ])
            if arm and len (Pk ):
                Dk =YO .uygula (Pk ,Dk ,cy7 .get (str (pid )),sk ,
                lambda Z :clf .predict_proba (Z )[:,1 ])
            if len (Pk )>1 :
                nm =wire_gate .crowd_mask (Pk ,sk )
                Pk ,Dk =Pk [nm ],Dk [nm ]
            tp ,fp ,fn =match_hungarian (Pk ,Dk ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
            signed =True )[:3 ]
            a =rob [r ["mfg"]]
            a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
            tes .append ((len (G ),)+match_hungarian (Pk ,Dk ,G ,Gd ,dg ,
            max (3.0 ,0.06 *dg ),180.0 ,True )[:3 ])
        pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
        mi =float (2 *sum (a [0 ]for a in rob .values ())/
        max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
        out [ad ]={"robot":mi ,"detection":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ())),"brand":pm }
        c =out [ad ]
        print (f"{ad :<24} robot {mi :.4f} | detection {c ['detection']:.4f} | makro "
        f"{c ['makro']:.4f} | en kotu {c ['en_kotu']:.4f}",flush =True )
    a ,b =out ["TEZ-SAF (direction degismez)"],out ["YON ODUNC"]
    art =sum (1 for m in b ["brand"]if b ["brand"][m ]>a ["brand"][m ]+1e-9 )
    yik =[m for m in b ["brand"]if b ["brand"][m ]==0 and a ["brand"][m ]>0 ]
    print (f"\nFARK robot {b ['robot']-a ['robot']:+.4f} | detection "
    f"{b ['detection']-a ['detection']:+.4f} | artan brand {art }/{len (b ['brand'])}"
    +(f" | YIKILAN: {','.join (yik )}"if yik else ""))
    for m in sorted (a ["brand"],key =lambda x :-a ["brand"][x ]):
        print (f"  {m :<7} {a ['brand'][m ]:.4f} -> {b ['brand'][m ]:.4f}  "
        f"{b ['brand'][m ]-a ['brand'][m ]:+.4f}")
    json .dump ({"damga":receipt_hash .damga (),"sonuc":out ,"artan_marka":art ,
    "n_secenek_egitim":int (len (X )),"pozitif":float (Y .mean ()),
    "not":"Konum DEGISMEZ, yalnizca direction secilir -> candidate sayisi ve FP "
    "riski artmaz. D7 brand-disi, MIKRO. Tez `v_o` konumu ve "
    "yonu 0. option, esitlikte kazanir."},
    open ("results/direction_borrow_d7.json","w"),indent =1 )
    print ("receipt -> results/direction_borrow_d7.json")


if __name__ =="__main__":
    main ()
