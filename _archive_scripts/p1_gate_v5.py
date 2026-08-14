# -*- coding: utf-8 -*-
"""P1-GATE-B: gate v5'i egit and TEMIZ sinavda olc.

MERDIVEN (results/d6_ceiling_ladder.json) gate'i EN BUYUK odul showed:
mevcut adaylarla mukemmel gate tespiti 0.4479 -> 0.9134 yapiyordu (+0.4655).

UC KOL, all of them AYNI temiz sinavda (468 part / 8 manufacturer, none of them egitimde YOK):
  TABAN    : dagitilan `results/wire_gate.pkl` (bugunku urun)          -> detection 0.4479
  v5-DUZ   : new korpusun tamami (34986 candidate / 2584 part / 9 manufacturer)
  v5-DENGE : manufacturer basina TAVAN uygulanmis corpus

DENGE WHY OLCULUYOR: new korpusun **%49.4'u TOGI**. Gate TOGI'ye ozellesirse
unseen ureticide coker -- [[gate-manufacturer-disi-cokusu]] full this desendi. G2
(manufacturer-dengeli) more before NULL cikmisti but that zamanki bilesim bambaskaydi;
single a manufacturer korpusun yarisi degildi.

KILL: temiz sinavda detection 0.4479'u GECMEZSE dagitma, `wire_gate.pkl` DOKUNULMAZ.
"""
import argparse 
import collections 
import glob 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

import d6_record 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

V3 ="results/zengin_parite_v3.npz"
KUME ="results/d6_exam_set.json"
MAKBUZ ="results/p1_gate_v5.json"
TABAN_TESPIT =0.4479 # dagitilan gate, same kumede measured
ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 


def egit (X ,y ,pid ,seed =0 ):
    """DAGITILAN gate with AYNI recete: RF 400 / leaf 3, ham -> part-ici z-score."""
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ),float )
    for p in np .unique (pid ):
        m =pid ==p 
        Z [m ]=wire_gate .within_part (X [m ],"zskor")
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =seed )
    clf .fit (Z ,y )
    return {"clf":clf ,"n_feat":Z .shape [1 ],"donusum":"zskor","cols":None ,
    "feat_names":None }


def dengele (mfg ,pid ,tavan_pay =0.25 ,rng =0 ):
    """Hicbir manufacturer korpusun `tavan_pay` kadarindan fazlasini kaplamasin.

    Aday not PARCA duzeyinde ornekler -- part-ici z-score parcayi BOLUNMEZ kilar.
    """
    r =np .random .RandomState (rng )
    parca_mfg ={}
    for m ,p in zip (mfg ,pid ):
        parca_mfg .setdefault (p ,m )
    parts =collections .defaultdict (list )
    for p ,m in parca_mfg .items ():
        parts [m ].append (p )
    total_ =len (parca_mfg )
    ceiling =max (1 ,int (tavan_pay *total_ ))
    tut =set ()
    for m ,ps in parts .items ():
        ps =sorted (ps )
        if len (ps )>ceiling :
            ps =list (r .choice (ps ,ceiling ,replace =False ))
        tut |=set (ps )
    return np .isin (pid ,sorted (tut ))


def olc (model ,rec_ ,match_greedy ,f1w ):
    import wire_gate 
    T ,R =[],[]
    Tm =collections .defaultdict (list )
    for pid ,r in rec_ .items ():
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="very"if r ["n"]>=8 else "low"
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        if r .get ("X")is not None and r .get ("P")is not None and len (r ["P"]):
            M =d6_record .x58 (r )
            if M is not None and M .shape [1 ]*2 ==model ["n_feat"]:
                k =wire_gate .decision_mask (wire_gate .decision_score (model ,M ))
                if k .any ():
                    P =np .asarray (r ["P"],float )[k ];D =np .asarray (r ["Pd"],float )[k ]
        t =match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ]
        T .append ((rj ,)+t );Tm [r ["mfg"]].append ((rj ,)+t )
        R .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,ROBOT_ACI ,
        False ,signed =True )[:3 ])
    return f1w (T ),f1w (R ),{m :f1w (v )for m ,v in Tm .items ()}


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ceiling-pay",type =float ,default =0.25 )
    a =ap .parse_args ()
    import protocol 
    protocol .tez_dogrula ()
    from sina_cluster import match_greedy ,f1w 

    sv =json .load (io .open (KUME ,encoding ="utf-8"))
    PID =set (sv ["pidler"])
    import d6_record 
    rec_ =d6_record .yukle (PID )
    print (f"TEMIZ SINAV: {len (rec_ )} part | muhur {sv ['sha16']}\n")

    d =np .load (V3 ,allow_pickle =True )
    X =np .hstack ([d ["X22"],d ["XR"]]).astype (float )
    y =np .asarray (d ["y"]);pid =np .asarray (d ["pids"],str );mfg =np .asarray (d ["mfg"],str )

    kollar ={}
    with open ("results/wire_gate.pkl","rb")as f :
        kollar ["TABAN (dagitilan)"]=pickle .load (f )
    print ("v5-DUZ egitiliyor...",flush =True )
    kollar ["v5-DUZ"]=egit (X ,y ,pid )
    msk =dengele (mfg ,pid ,a .tavan_pay )
    up =collections .Counter (mfg [msk ])
    print (f"v5-DENGE egitiliyor (manufacturer tavani %{100 *a .tavan_pay :.0f}) -- "
    f"{msk .sum ()} candidate / {len (np .unique (pid [msk ]))} part",flush =True )
    print (f"  dagilim: {dict (up .most_common (6 ))}")
    kollar ["v5-DENGE"]=egit (X [msk ],y [msk ],pid [msk ])

    print (f"\n{'arm':<20}{'TESPIT':>9}{'ROBOT':>9}{'kill':>8}")
    res_ ={}
    for ad ,m in kollar .items ():
        tf ,rf ,um =olc (m ,rec_ ,match_greedy ,f1w )
        kill =""if ad .startswith ("TABAN")else ("GECTI"if tf >TABAN_TESPIT else "KALDI")
        print (f"{ad :<20}{tf :>9.4f}{rf :>9.4f}{kill :>8}")
        res_ [ad ]={"detection":tf ,"robot":rf ,"manufacturer":um }

    print (f"\n{'manufacturer':<8}"+"".join (f"{k [:12 ]:>14}"for k in kollar ))
    urs =sorted ({u for k in res_ for u in res_ [k ]["manufacturer"]})
    for u in urs :
        print (f"{u :<8}"+"".join (f"{res_ [k ]['manufacturer'].get (u ,float ('nan')):>14.4f}"
        for k in kollar ))

    en_iyi =max ((k for k in kollar if not k .startswith ("TABAN")),
    key =lambda k :res_ [k ]["detection"])
    print (f"\nEN IYI YENI KOL: {en_iyi } detection {res_ [en_iyi ]['detection']:.4f} "
    f"(baseline {TABAN_TESPIT :.4f}, diff {res_ [en_iyi ]['detection']-TABAN_TESPIT :+.4f})")
    if res_ [en_iyi ]["detection"]>TABAN_TESPIT :
        with open ("results/wire_gate_v5.pkl","wb")as f :
            pickle .dump (kollar [en_iyi ],f )
        print ("  -> results/wire_gate_v5.pkl yazildi (NOT DEPLOYED; wire_gate.pkl dokunulmadi)")
    else :
        print ("  -> KILL: no arm tabani gecmedi, model YAZILMADI")
    json .dump ({k :{"detection":v ["detection"],"robot":v ["robot"],"manufacturer":v ["manufacturer"]}
    for k ,v in res_ .items ()}|{"taban_tespit":TABAN_TESPIT ,
    "muhur":sv ["sha16"],"tavan_pay":a .tavan_pay },
    io .open (MAKBUZ ,"w",encoding ="utf-8"),indent =1 )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
