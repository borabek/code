# -*- coding: utf-8 -*-
"""v6 FARKI, ADIM A: difference KORPUS/ETIKET'ten mi geliyor?

DURUM: dagitilan gate v6 tez-saf havuzda D7'de **0.1970**; same receteyle
(within_part zskor PARCA PARCA, RF 400/leaf3, urunun `decision_score`'this) benim
kurdugum gate same kumede **0.1159**. 0.08'lik difference kapanmadan genisletilmis
havuzun olculmus +0.0386'si gerceklesmiyor.

ADIM A: same receteyi v6'nin KENDI korpusu (`zengin_parite_v6.npz`,
53036 candidate / 2583 part, pozitif 0.2613) on kosarim.
  ~0.1970 cikarsa  -> difference KORPUS/ETIKET tanimindadir
  ~0.1159 kalirsa  -> difference `_cokus_yonlendir` ya da feature kaynagindadir

SIZINTI DENETIMI YAPILDI: korpusun D7 with kesisimi PARCA 0, MARKA 0.

Olcum: `_brep_oz` onbellegindeki D7 TEZ-SAF adaylari (source==0), urunun karar
kurali + NMS, MIKRO. Bu, 0.1970 and 0.1159'un olculdugu KUMENIN AYNISI.
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
import canonical_d7 as K # noqa: E402
import wire_gate # noqa: E402
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OZ ="results/_brep_oz"
DONUSUM ="zskor"


def korpustan_egit (path ):
    """refit_gate_v7 recetesi: within_part PARCA PARCA + RF 400/leaf3."""
    d =np .load (path ,allow_pickle =True )
    X =np .hstack ([d ["X22"],d ["XR"]]).astype (float )
    y =np .asarray (d ["y"]).astype (int )
    pids =np .array ([str (p )for p in d ["pids"]])
    M =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pids ):
        i =np .where (pids ==u )[0 ]
        M [i ]=wire_gate .within_part (X [i ],DONUSUM )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M ,y )
    return ({"clf":clf ,"cols":None ,"n_feat":M .shape [1 ],"donusum":DONUSUM },
    M .shape ,float (y .mean ()),len (np .unique (pids )))


def exam ():
    te =[]
    for f in sorted (os .listdir (OZ )):
        if not (f .startswith ("d7_")and f .endswith (".npz")):
            continue 
        z =np .load (f"{OZ }/{f }")
        if "source"not in z :
            continue 
        m =z ["source"]==0 # TEZ-SAF pool
        te .append ({"pid":f [3 :-4 ],"X":np .asarray (z ["X"],float )[m ],
        "P":z ["P"][m ],"D":z ["D"][m ]})
    kay =K .yukle ([d ["pid"]for d in te ])
    for d in te :
        r =kay [d ["pid"]]
        d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
        "Gd":np .asarray (r ["Gd"],float ),"diag":r ["diag"]})
    return te 


def olc (model ,te ):
    en =None 
    for tip ,e in ([("mutlak",x )for x in (0.20 ,0.30 ,0.40 )]+
    [("goreli",x )for x in ((0.5 ,0.20 ),(0.5 ,0.30 ))]):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
        tes =[]
        for d in te :
            X =d ["X"]
            if len (X )<2 :
                continue 
            s =np .asarray (wire_gate .decision_score (model ,X ),float )
            k =(s >=e )if tip =="mutlak"else maske (s ,e [0 ],e [1 ])
            P ,D =(d ["P"][k ],d ["D"][k ])if k .any ()else (d ["P"][:0 ],d ["D"][:0 ])
            if len (P )>1 :
                nm =wire_gate .crowd_mask (P ,s [k ])
                P ,D =P [nm ],D [nm ]
            tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
            K .ACI ,False ,signed =True )[:3 ]
            a =rob [d ["mfg"]]
            a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
            tes .append ((len (d ["G"]),)+match_hungarian (
            P ,D ,d ["G"],d ["Gd"],d ["diag"],
            max (3.0 ,0.06 *d ["diag"]),180.0 ,True )[:3 ])
        pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
        mi =float (2 *sum (a [0 ]for a in rob .values ())/
        max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
        r ={"rule":f"{tip } {e }","robot":mi ,"detection":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ()))}
        if en is None or r ["robot"]>en ["robot"]:
            en =r 
    return en 


def main ():
    te =exam ()
    print (f"SINAV {len (te )} part (D7 tez-saf pool)\n",flush =True )
    out ={}
    g6 =K .gate_yukle ()
    out ["v6 (DAGITILAN)"]=olc (g6 ,te )
    print (f"{'v6 (DAGITILAN)':<34} robot {out ['v6 (DAGITILAN)']['robot']:.4f} | "
    f"detection {out ['v6 (DAGITILAN)']['detection']:.4f} | "
    f"{out ['v6 (DAGITILAN)']['rule']}",flush =True )
    for path in ("results/zengin_parite_v6.npz","results/zengin_parite_v3.npz"):
        ad =f"benim recete + {os .path .basename (path )}"
        m ,sh ,poz ,np_ =korpustan_egit (path )
        out [ad ]=olc (m ,te )
        c =out [ad ]
        print (f"{ad :<34} robot {c ['robot']:.4f} | detection {c ['detection']:.4f} | "
        f"{c ['rule']} | training {sh } part {np_ } pozitif {poz :.4f}",flush =True )
    v =out ["v6 (DAGITILAN)"]["robot"]
    b =out ["benim recete + zengin_parite_v6.npz"]["robot"]
    print (f"\nBENIM KORPUSUMLA (onceki measurement): 0.1159")
    print (f"v6 KORPUSUYLA:                   {b :.4f}")
    print (f"v6 KENDISI:                      {v :.4f}")
    print ("\nTESHIS: "+("diff KORPUS/ETIKET tanimindan -- kapatilabilir"
    if b >=v -0.02 else 
    "corpus ACIKLAMIYOR -- diff `_cokus_yonlendir` ya da "
    "oznitelik kaynaginda, ADIM B'ye gecilir"))
    json .dump ({"damga":receipt_hash .damga (),"sonuc":out ,
    "benim_korpusum_robot":0.1159 ,
    "not":"ADIM A: same recete, FARKLI corpus. D7 tez-saf pool, MIKRO. "
    "Sizinti denetimi: corpus-D7 kesisimi part 0, brand 0."},
    open ("results/v6_delta_A.json","w"),indent =1 )
    print ("receipt -> results/v6_delta_A.json")


if __name__ =="__main__":
    main ()
