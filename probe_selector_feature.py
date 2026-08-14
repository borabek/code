# -*- coding: utf-8 -*-
"""SECICI VERIMI: genisletilmis havuzda hangi EK feature whereas yariyor?

Havuz acildi (robot tavani 0.5748) but selector tavanin ~%35'ini yakaliyor. Bu
probe, YENIDEN HESAP GEREKTIRMEYEN ek oznitelikleri onbellekli veriyle dener:
  K) KAYNAK   -- candidate segmentasyondan mi B-rep'ten mi geldi (1 column)
  Y) YOGUNLUK -- most yakin komsuya uzaklik, 5/10mm yaricapta komsu count (3)
  M) MERKEZ   -- part kutusuna according to konum + eksenlere hizalanma (3)
Egitim D6 (468), exam D7 (835) -- brand kumeleri AYRIK. Tek degisken, same
threshold taramasi, same RF ayarlari.

WARNING: this D6-olcekli a kiyas; mutlak degerler full olcekli gate'ten DUSUK becomes.
Aranan sey MUTLAK DEGER DEGIL, hangi feature grubunun ISE YARADIGI.
"""
import collections ,json ,os ,sys 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import wire_gate ,canonical_d7 as K ,d6_record 
from sklearn .ensemble import RandomForestClassifier 
from sina_cluster import match_hungarian 

OZ ="results/_brep_oz"


def yukle (on ):
    v =[]
    for f in sorted (os .listdir (OZ )):
        if f .startswith (on +"_")and f .endswith (".npz"):
            z =np .load (f"{OZ }/{f }")
            if "P"not in z :
                continue 
            v .append ({"pid":f [len (on )+1 :-4 ],"X":z ["X"],"y":z ["y"],
            "P":z ["P"],"D":z ["D"],"kaynak":z ["kaynak"]})
    return v 


def ek (d ,kul ):
    """Yeniden hesap GEREKTIRMEYEN ek sutunlar."""
    P =d ["P"];n =len (P );sut =[]
    if "K"in kul :
        sut .append (d ["kaynak"].reshape (-1 ,1 ).astype (float ))
    if "Y"in kul :
        if n >1 :
            M =np .linalg .norm (P [:,None ]-P [None ],axis =-1 )
            np .fill_diagonal (M ,np .inf )
            en =M .min (1 ).reshape (-1 ,1 )
            k5 =(M <5 ).sum (1 ).reshape (-1 ,1 ).astype (float )
            k10 =(M <10 ).sum (1 ).reshape (-1 ,1 ).astype (float )
        else :
            en =np .zeros ((n ,1 ));k5 =np .zeros ((n ,1 ));k10 =np .zeros ((n ,1 ))
        sut +=[en ,k5 ,k10 ]
    if "M"in kul :
        lo ,hi =P .min (0 ),P .max (0 )
        sp =np .maximum (hi -lo ,1e-6 )
        rel =(P -lo )/sp 
        sut .append (np .linalg .norm (rel -0.5 ,axis =1 ).reshape (-1 ,1 ))
        sut .append (np .abs (d ["D"]).max (1 ).reshape (-1 ,1 ))# eksene hizali mi
        sut .append (rel [:,np .argmax (sp )].reshape (-1 ,1 ))# uzun eksende yer
    return np .hstack ([d ["X"]]+sut )if sut else d ["X"]


tr ,te =yukle ("d6"),yukle ("d7")
k6 =d6_record .yukle ({d ["pid"]for d in tr })
for d in tr :
    d ["mfg"]=k6 [d ["pid"]]["mfg"]
k7 =K .yukle ([d ["pid"]for d in te ])
for d in te :
    r =k7 [d ["pid"]]
    d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
    "Gd":np .asarray (r ["Gd"],float ),"diag":r ["diag"]})
print (f"D6 {len (tr )} | D7 {len (te )}\n",flush =True )

sonuc ={}
for ad ,kul in (("baseline (58)",""),("+KAYNAK","K"),("+YOGUNLUK","Y"),
("+MERKEZ","M"),("+HEPSI","KYM")):
    X =np .vstack ([ek (d ,kul )for d in tr ])
    y =np .concatenate ([d ["y"]for d in tr ])
    c =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =2 ,n_jobs =-1 ,
    class_weight ="balanced_subsample",random_state =0 )
    c .fit (X ,y )
    en =None 
    for e in (0.20 ,0.30 ,0.40 ,0.50 ):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ]);tes =[]
        for d in te :
            s =c .predict_proba (ek (d ,kul ))[:,1 ]
            k =s >=e 
            P ,D =(d ["P"][k ],d ["D"][k ])if k .any ()else (d ["P"][:0 ],d ["D"][:0 ])
            if len (P )>1 :
                nm =wire_gate .crowd_mask (P ,s [k ]);P ,D =P [nm ],D [nm ]
            tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
            K .ACI ,False ,signed =True )[:3 ]
            a =rob [d ["mfg"]];a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
            tes .append ((len (d ["G"]),)+match_hungarian (P ,D ,d ["G"],d ["Gd"],
            d ["diag"],max (3.0 ,0.06 *d ["diag"]),180.0 ,True )[:3 ])
        mi =float (2 *sum (a [0 ]for a in rob .values ())/
        max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
        pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
        r ={"threshold":e ,"robot":mi ,"tespit":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ())))}
        if en is None or r ["robot"]>en ["robot"]:
            en =r 
    sonuc [ad ]=dict (en ,sutun =int (X .shape [1 ]))
    print (f"{ad :<12} sutun {X .shape [1 ]:>3} | robot {en ['robot']:.4f} | "
    f"tespit {en ['tespit']:.4f} | makro {en ['makro']:.4f} | threshold {en ['threshold']:.2f}",
    flush =True )
t =sonuc ["baseline (58)"]["robot"]
print ()
for ad in sonuc :
    if ad !="baseline (58)":
        print (f"  {ad :<12} {sonuc [ad ]['robot']-t :+.4f}")
json .dump ({"damga":makbuz_hash .damga (),"sonuc":sonuc ,
"not":"D6-olcekli kiyas (training 468). Mutlak degerler tam olcekten "
"DUSUK; aranan hangi oznitelik grubunun ise yaradigi. D7 brand-disi."},
open ("results/secici_oznitelik.json","w"),indent =1 )
