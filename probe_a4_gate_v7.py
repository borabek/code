# -*- coding: utf-8 -*-
"""A4: g10 + gate v7 REFIT -- uctan uca A/B (D6, 468 part).

UC KOL, AYNI PARCALARDA:
  1. g7  + gate v5  = DAGITILAN state (baseline)
  2. g10 + gate v5  = new network, ESKI gate (refit YAPILMAMIS hali -- low cikmasi
                      BEKLENIR; "two gate same dagilimda egitilmeli" dersi)
  3. g10 + gate v7  = new network + REFIT gate  <- olculmek istenen
Hata YUTULMAZ.
"""
import json ,os ,pickle ,sys ,glob 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record 
from probe_k65b_olc import olc 
from korpus_kimlik import step_kimlik as SK 

KOLLAR =[("g7 + gate v5 (DAGITILAN)","results/_p1_olasilik_g7","results/wire_gate_v5.pkl"),
("g10 + gate v5 (refit YOK)","results/_p1_olasilik_g10","results/wire_gate_v5.pkl"),
("g10 + gate v7 (REFIT)","results/_p1_olasilik_g10","results/wire_gate_v7.pkl")]

sv =d6_record .exam ();rec_ =d6_record .yukle (set (sv ["pidler"]))
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
ortak =None 
for _ ,ob ,_g in KOLLAR :
    p ={f [:-4 ]for f in os .listdir (ob )if f .endswith (".npz")}
    ortak =p if ortak is None else (ortak &p )
pidler =sorted (ortak &set (rec_ ))
print (f"ortak part: {len (pidler )}\n",flush =True )

res_ ={}
for ad ,ob ,gy in KOLLAR :
    gate =pickle .load (open (gy ,"rb"))
    t ,r =olc (pidler ,ob ,rec_ ,gate ,S )
    res_ [ad ]={"tespit":t ,"robot":r ,"cache":ob ,"gate":gy }
    print (f"{ad :<28} tespit {t :.4f} | robot {r :.4f}",flush =True )

baseline =res_ ["g7 + gate v5 (DAGITILAN)"]
for ad ,v in res_ .items ():
    v ["tespit_fark"]=v ["tespit"]-baseline ["tespit"]
    v ["robot_fark"]=v ["robot"]-baseline ["robot"]
print ("\nDAGITILANA GORE FARK:")
for ad ,v in res_ .items ():
    print (f"  {ad :<28} tespit {v ['tespit_fark']:+.4f} | robot {v ['robot_fark']:+.4f}")
json .dump ({"damga":makbuz_hash .damga (),"sonuc":res_ ,"n_parca":len (pidler ),
"not":"D6 = DEV. Muhurlu exam DEGIL."},
open ("results/a4_gate_v7.json","w"),indent =1 )
print ("\nmakbuz -> results/a4_gate_v7.json")
