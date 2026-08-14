# -*- coding: utf-8 -*-
"""D7'de p5-v2, KANONIK girdilerle and MIKRO toplamayla.

GIRDILER (makbuzlardan dogrulandi): kayitlar `_der_yeni_G7BIRLESIK.pkl`,
gate `wire_gate_v6.pkl`. TOPLAMA: MIKRO (headline olcegi).
TABAN (same girdilerle measured): detection 0.2988 / robot 0.1956.

p5-v2 D6'da egitilir, D7'de olculur -- brand kumeleri AYRIK.
D7 = DEV. FINAL DEGIL.
"""
import collections ,glob ,json ,os ,pickle ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,wire_gate ,canonical_d7 as K ,p5v2_secenek as PS ,p5v2_egit as PE 
from p1c_threshold import maske 
from sina_cluster import match_hungarian 
from corpus_identity import step_kimlik as SK 

gate =K .gate_yukle ();S =K .step_map ()
cy6 =pickle .load (open ("results/_d6_silindirler.pkl","rb"))
ac6 =pickle .load (open ("results/_d6_acikliklar.pkl","rb"))
cy7 =pickle .load (open ("results/_d7_silindirler.pkl","rb"))
ac7 =pickle .load (open ("results/_d7_acikliklar.pkl","rb"))
d7p =json .load (open ("results/d7_exam_set.json"))["pidler"]
k7 =K .yukle (d7p )
d6 =d6_record .exam ();k6 =d6_record .yukle (set (d6 ["pidler"]))
print (f"D6 training {len (k6 )} | D7 measurement {len (k7 )}",flush =True )


def kur (rec_ ,cy ,ac ,etiketli ,x58f ):
    v =[]
    for pid ,r in rec_ .items ():
        X =x58f (r )
        P =np .asarray (r ["P"],float );D =np .asarray (r ["Pd"],float )
        if X is None or not len (P ):
            continue 
        G =np .asarray (r .get ("G",[]),float )
        if etiketli and not len (G ):
            continue 
        gs =np .asarray (wire_gate .decision_score (gate ,X ),float )
        komsu =None 
        if len (D )>1 :
            B =D *np .sign (D @D [0 ])[:,None ]
            komsu =B .mean (0 );komsu /=(np .linalg .norm (komsu )+1e-12 )
        secs =PS .options (P ,D ,cy .get (pid ),ac .get (pid ),r ["diag"],
        gate_s =gs ,komsu =komsu )
        d ={"pid":pid ,"mfg":r ["mfg"],"secs":secs ,"gate_skor":gs ,"G":G ,
        "Gd":np .asarray (r .get ("Gd",[]),float ),"diag":r ["diag"],
        "P":P ,"D":D }
        if etiketli :
            d ["y"]=PE .etiketle (secs ,G ,d ["Gd"])
        v .append (d )
    return v 


print ("D6 training...",flush =True )
tr =kur (k6 ,cy6 ,ac6 ,True ,d6_record .x58 )
clf ,sh ,poz =PE .egit (tr )
print (f"  {len (tr )} part | {sh } | pozitif {poz :.4f}",flush =True )
te =kur (k7 ,cy7 ,ac7 ,False ,K .x58 )
print (f"D7 measurement {len (te )} part",flush =True )

out ={}
for ad ,p5 in (("TABAN (gate -> tam_poz)",False ),("p5-v2 (ortak secim -> gate)",True )):
    rows =[];per =collections .defaultdict (lambda :[0 ,0 ,0 ])
    for d in te :
        if not len (d ["G"]):
            continue 
        if p5 :
            score =[clf .predict_proba (np .asarray ([s [2 ]for s in o ],float ))[:,1 ]
            for o in d ["secs"]]
            P ,D =PE .sec (d ["secs"],score ,gate_skor =d ["gate_skor"],
            gate_esik =(0.40 ,0.30 ))
        else :
            k =maske (d ["gate_skor"],0.40 ,0.30 )
            P ,D =(d ["P"][k ],d ["D"][k ])if k .any ()else (d ["P"][:0 ],d ["D"][:0 ])
        tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
        K .YANAL ,K .ACI ,False ,signed =True )[:3 ]
        rows .append ((len (d ["G"]),tp ,fp ,fn ))
        a =per [d ["mfg"]];a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
    mi ,ma =K .mikro (rows ),K .makro (per )
    ku =min (2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for a in per .values ())
    out [ad ]={"mikro":mi ,"makro":ma ,"en_kotu":ku }
    print (f"{ad :<28} MIKRO {mi :.4f} | makro {ma :.4f} | en kotu {ku :.4f}",flush =True )
a ,b =list (out .values ())
print (f"\nFARK: mikro {b ['mikro']-a ['mikro']:+.4f} | makro {b ['makro']-a ['makro']:+.4f}")
json .dump ({"damga":receipt_hash .damga (),"sonuc":out ,"n_parca":len (te ),
"not":"KANONIK girdiler (G7BIRLESIK + gate v6), MIKRO toplama. D7=DEV."},
open ("results/d7_p5v2_canonical.json","w"),indent =1 )
print ("receipt -> results/d7_p5v2_canonical.json")
