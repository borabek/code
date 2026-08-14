# -*- coding: utf-8 -*-
"""MADDE 6: same FIZIKSEL AGIZA yigilan adaylari bastirmak F1'i artirir mi?

Urun zincirinde two candidate same silindire/agza dusuyorsa ikisi de cikiyor; Macar
bire-a eslestirdigi for biri zorunlu FP. p5-v2 bunu `agiz_kimlik` uzerinden
bipartite with cozuyordu; here AYNI kisiti URUN zincirine single basina, a
NMS as uyguluyoruz (each agizdan most high gate skorlusu kalir).

Uc radius denenir; also mouth kimligi instead of SAF distance-NMS'i de olculur.
KANONIK girdiler, MIKRO, D7 (=DEV).
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import wire_gate ,canonical_d7 as K 
from p1c_threshold import maske 
from sina_cluster import match_hungarian 

gate =K .gate_yukle ()
cy7 =pickle .load (open ("results/_d7_silindirler.pkl","rb"))
k7 =K .yukle (json .load (open ("results/d7_exam_set.json"))["pidler"])


def agiz_kimlikleri (P ,cyl ,R ):
    """Her adayi most yakin FIZIKSEL AGIZA baglar; otherwise -1 (bastirilmaz).

    Silindir kaydinin anahtarlari INGILIZCE: center/axis/radius/mouth_a/mouth_b.
    Kimlik silindir DEGIL AGIZ duzeyinde: same silindirin two ucu AYRI agizdir,
    ikisine de birer candidate dusebilir and this kalabalik SAYILMAZ.
    """
    kim =np .full (len (P ),-1 )
    if not cyl :
        return kim 
    A =[]
    for c in cyl :
        for u in ("mouth_a","mouth_b"):
            if c .get (u )is not None :
                A .append (np .asarray (c [u ],float ))
    if not A :
        return kim 
    A =np .asarray (A ,float )
    for i ,p in enumerate (P ):
        d =np .linalg .norm (A -p ,axis =1 )
        j =int (np .argmin (d ))
        if d [j ]<=R :
            kim [i ]=j 
    return kim 


def nms_kimlik (P ,gs ,cyl ,R ):
    kim =agiz_kimlikleri (P ,cyl ,R )
    tut =np .ones (len (P ),bool )
    for j in set (kim .tolist ())-{-1 }:
        idx =np .where (kim ==j )[0 ]
        if len (idx )>1 :
            tut [idx ]=False 
            tut [idx [int (np .argmax (gs [idx ]))]]=True 
    return tut 


def nms_mesafe (P ,gs ,r ):
    rank_ =np .argsort (-gs );tut =np .ones (len (P ),bool )
    for a in range (len (rank_ )):
        i =rank_ [a ]
        if not tut [i ]:
            continue 
        for b in range (a +1 ,len (rank_ )):
            j =rank_ [b ]
            if tut [j ]and np .linalg .norm (P [i ]-P [j ])<r :
                tut [j ]=False 
    return tut 


arm =collections .defaultdict (list )
for pid ,r in k7 .items ():
    X =K .x58 (r );G =np .asarray (r .get ("G",[]),float )
    if X is None or not len (G ):
        continue 
    P =np .asarray (r ["P"],float );D =np .asarray (r ["Pd"],float )
    Gd =np .asarray (r ["Gd"],float );dg =r ["diag"]
    gs =np .asarray (wire_gate .decision_score (gate ,X ),float )
    m =maske (gs ,0.40 ,0.30 )
    if not m .any ():
        for ad in ("threshold (urun)",):
            arm [ad ].append ((len (G ),0 ,0 ,len (G )))
        continue 
    Pm ,Dm ,gm =P [m ],D [m ],gs [m ]
    cyl =cy7 .get (pid )

    def ol (sel ):
        tp ,fp ,fn =match_hungarian (Pm [sel ],Dm [sel ],G ,Gd ,dg ,K .YANAL ,K .ACI ,
        False ,signed =True )[:3 ]
        return (len (G ),tp ,fp ,fn )

    arm ["threshold (urun)"].append (ol (np .ones (len (Pm ),bool )))
    for R in (2.0 ,4.0 ,6.0 ):
        arm [f"mouth-NMS R={R }"].append (ol (nms_kimlik (Pm ,gm ,cyl ,R )))
    for rr in (1.5 ,3.0 ):
        arm [f"mesafe-NMS r={rr }"].append (ol (nms_mesafe (Pm ,gm ,rr )))

baseline =K .mikro (arm ["threshold (urun)"])
print (f"D7 {len (arm ['threshold (urun)'])} part\n")
res ={}
for ad in arm :
    res [ad ]=K .mikro (arm [ad ])
    d =res [ad ]-baseline 
    print (f"  {ad :<18} MIKRO {res [ad ]:.4f}  {d :+.4f}")
en =max ((a for a in res if a !="threshold (urun)"),key =lambda a :res [a ])
print (f"\nEN IYI: {en } {res [en ]-baseline :+.4f}")
print ("DECISION: "+("madde 6 ACIK"if res [en ]-baseline >=0.01 else "madde 6 OLU"))
json .dump ({"damga":receipt_hash .damga (),"mikro":res ,"baseline":baseline ,
"en_iyi":en ,"kazanc":res [en ]-baseline ,"n":len (arm ["threshold (urun)"]),
"not":"URUN zincirine tek basina NMS. D7=DEV. MIKRO."},
open ("results/suppress_crowd.json","w"),indent =1 )
