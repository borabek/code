# -*- coding: utf-8 -*-
"""CHECK: genisletilmis pool mu kotu, otherwise D6-single gate egitimi mi small?

`run_brep_gate.py` genisletilmis havuzda robot 0.1568 verdi; kanonik 0.2029.
Ama two arm AYNI SEYI degistirmedi: refit gate only D6'nin 468 parcasiyla
egitildi, kanonik gate v6 binlerce parcayla. Karistirici ayrilmadan "pool kotu"
DENEMEZ.

Bu betik same onbellekli ozniteliklerle UC kolu same training buyuklugunde kiyaslar:
  A) SEG-TEK pool  + D6-single gate     <- kontrol
  B) GENISLETILMIS  + D6-single gate     <- deney
  C) SEG-TEK pool  + kanonik gate v6 <- referans (bilinen 0.2029)
A with B arasindaki difference HAVUZUN etkisidir; A with C arasindaki difference EGITIM
BUYUKLUGUNUN etkisidir.
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import wire_gate ,canonical_d7 as K 
from sklearn .ensemble import RandomForestClassifier 
from sina_cluster import match_hungarian 

OZ ="results/_brep_oz"


def yukle (ad ):
    v =[]
    for f in sorted (os .listdir (OZ )):
        if not f .startswith (ad +"_")or not f .endswith (".npz"):
            continue 
        z =np .load (f"{OZ }/{f }")
        v .append ({"pid":f [len (ad )+1 :-4 ],"X":z ["X"],"y":z ["y"],"P":z ["P"],
        "D":z ["D"],"source":z ["source"]})
    return v 


tr ,te =yukle ("d6"),yukle ("d7")
kay7 =K .yukle ([d ["pid"]for d in te ])
kay6 =None 
import d6_record 
kay6 =d6_record .yukle ({d ["pid"]for d in tr })
for d in tr :
    r =kay6 [d ["pid"]];d ["mfg"]=r ["mfg"]
for d in te :
    r =kay7 [d ["pid"]]
    d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
    "Gd":np .asarray (r ["Gd"],float ),"diag":r ["diag"]})
print (f"D6 {len (tr )} | D7 {len (te )}",flush =True )


def egit (segtek ):
    X ,y =[],[]
    for d in tr :
        m =d ["source"]==0 if segtek else np .ones (len (d ["y"]),bool )
        X .append (d ["X"][m ]);y .append (d ["y"][m ])
    X =np .vstack (X );y =np .concatenate (y )
    c =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =2 ,n_jobs =-1 ,
    class_weight ="balanced_subsample",random_state =0 )
    c .fit (X ,y )
    return c ,X .shape ,float (y .mean ())


def olc (skorla ,segtek ,esikler ):
    en =None 
    for e in esikler :
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ]);tes =[]
        for d in te :
            m0 =d ["source"]==0 if segtek else np .ones (len (d ["y"]),bool )
            s =skorla (d ["X"][m0 ]);P ,D =d ["P"][m0 ],d ["D"][m0 ]
            k =s >=e 
            P2 ,D2 =(P [k ],D [k ])if k .any ()else (P [:0 ],D [:0 ])
            if len (P2 )>1 :
                nm =wire_gate .crowd_mask (P2 ,s [k ]);P2 ,D2 =P2 [nm ],D2 [nm ]
            tp ,fp ,fn =match_hungarian (P2 ,D2 ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
            K .ACI ,False ,signed =True )[:3 ]
            a =rob [d ["mfg"]];a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
            tes .append ((len (d ["G"]),)+match_hungarian (P2 ,D2 ,d ["G"],d ["Gd"],
            d ["diag"],max (3.0 ,0.06 *d ["diag"]),180.0 ,True )[:3 ])
        mi =float (2 *sum (a [0 ]for a in rob .values ())/
        max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
        pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
        r ={"threshold":e ,"robot":mi ,"tespit":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ()))}
        if en is None or r ["robot"]>en ["robot"]:
            en =r 
    return en 


ES =(0.20 ,0.30 ,0.40 ,0.50 )
out ={}
for ad ,segtek in (("A) SEG-TEK + D6 gate",True ),
("B) GENISLETILMIS + D6 gate",False )):
    c ,sh ,poz =egit (segtek )
    out [ad ]=olc (lambda X :c .predict_proba (X )[:,1 ],segtek ,ES )
    r =out [ad ]
    print (f"{ad :<28} robot {r ['robot']:.4f} | tespit {r ['tespit']:.4f} | "
    f"makro {r ['makro']:.4f} | threshold {r ['threshold']:.2f} | training {sh } poz {poz :.4f}",
    flush =True )
g6 =K .gate_yukle ()
out ["C) SEG-TEK + kanonik gate v6"]=olc (
lambda X :np .asarray (wire_gate .decision_score (g6 ,X ),float ),True ,ES )
r =out ["C) SEG-TEK + kanonik gate v6"]
print (f"{'C) SEG-TEK + kanonik v6':<28} robot {r ['robot']:.4f} | tespit {r ['tespit']:.4f} | "
f"makro {r ['makro']:.4f} | threshold {r ['threshold']:.2f}",flush =True )
A ,B ,C =(out ["A) SEG-TEK + D6 gate"],out ["B) GENISLETILMIS + D6 gate"],
out ["C) SEG-TEK + kanonik gate v6"])
print (f"\nHAVUZUN etkisi (B-A): robot {B ['robot']-A ['robot']:+.4f} | "
f"tespit {B ['tespit']-A ['tespit']:+.4f}")
print (f"EGITIM BUYUKLUGUNUN etkisi (A-C): robot {A ['robot']-C ['robot']:+.4f} | "
f"tespit {A ['tespit']-C ['tespit']:+.4f}")
json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,
"havuz_etkisi":B ["robot"]-A ["robot"],
"egitim_buyuklugu_etkisi":A ["robot"]-C ["robot"],
"not":"Ayni onbellekli oznitelikler, same threshold taramasi. D7 brand-disi, MIKRO."},
open ("results/brep_kontrol.json","w"),indent =1 )
