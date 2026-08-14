# -*- coding: utf-8 -*-
"""D7 MANSET YOLUNDA p5-v2 -- 0.2344 with AYNI OLCEKTE single number.

MANSET YOLU (0.2344'u ureten): kayittaki candidates (r["P"], r["Pd"]) -> gate maskesi
-> `product_chain.tam_poz` -> Macar. Bu betik AYNI yolu kullanir, single difference
p5-v2'nin araya girmesi.

  A) MANSET: gate maskesi -> tam_poz                      (beklenen ~0.2344)
  B) p5-v2 : HAM pool -> ortak secim -> gate SONRA        (olculmek istenen)

EGITIM/OLCUM AYRIMI: p5-v2 D6'nin 8 markasinda egitilir, D7'nin 12 markasinda
olculur -- kumeler AYRIK. D7 = DEV (harcandi), this FINAL DEGIL.
TEZE SADIK: `v_o` each two kolda da havuzda and p5-v2'de GERCEK fallback.
"""
import glob ,json ,os ,pickle ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,wire_gate ,product_chain ,p5v2_secenek as PS ,p5v2_egit as PE 
from p1c_threshold import maske 
from sina_cluster import match_hungarian ,f1w 
from corpus_identity import step_kimlik as SK 

YANAL ,ACI =2.0 ,10.0 
gate =pickle .load (open ("results/wire_gate_v5.pkl","rb"))
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
OB ="results/_p1_olasilik_d7"# D7 onbellegi -- D6 verirsem tam_poz HIC KOSMAZ

d6 =d6_record .exam ()
k6 =d6_record .yukle (set (d6 ["pidler"]))
d7p =set (map (str ,json .load (open ("results/d7_exam_set.json"))["pidler"]))
k7 =d6_record .yukle (d7p )
cy6 =pickle .load (open ("results/_d6_silindirler.pkl","rb"))
ac6 =pickle .load (open ("results/_d6_acikliklar.pkl","rb"))
cy7 =pickle .load (open ("results/_d7_silindirler.pkl","rb"))
ac7 =pickle .load (open ("results/_d7_acikliklar.pkl","rb"))
print (f"D6 training {len (k6 )} part | D7 measurement {len (k7 )} part",flush =True )


def kur (rec_ ,cy ,ac ,etiketli ):
    data_ =[]
    for pid ,r in rec_ .items ():
        X =d6_record .x58 (r )
        if X is None or r .get ("P")is None or not len (r ["P"]):
            continue 
        G =np .asarray (r .get ("G",[]),float )
        if etiketli and not len (G ):
            continue 
        P =np .asarray (r ["P"],float );D =np .asarray (r ["Pd"],float )
        gs =np .asarray (wire_gate .decision_score (gate ,X ),float )
        komsu =None 
        if len (D )>1 :
            B =D *np .sign (D @D [0 ])[:,None ]
            komsu =B .mean (0 );komsu /=(np .linalg .norm (komsu )+1e-12 )
        secs =PS .options (P ,D ,cy .get (pid ),ac .get (pid ),r ["diag"],
        gate_s =gs ,komsu =komsu )
        d ={"pid":pid ,"mfg":r ["mfg"],"secs":secs ,"gate_skor":gs ,
        "G":G ,"Gd":np .asarray (r .get ("Gd",[]),float ),"diag":r ["diag"],
        "P":P ,"D":D ,"X":X }
        if etiketli :
            d ["y"]=PE .etiketle (secs ,G ,d ["Gd"])
        data_ .append (d )
    return data_ 


print ("D6 training verisi kuruluyor...",flush =True )
tr =kur (k6 ,cy6 ,ac6 ,True )
clf ,sh ,poz =PE .egit (tr )
print (f"  training {len (tr )} part | {sh } | pozitif {poz :.4f}",flush =True )
print ("D7 measurement verisi kuruluyor...",flush =True )
te =kur (k7 ,cy7 ,ac7 ,False )
print (f"  measurement {len (te )} part",flush =True )

# A) MANSET KOLU
T ,R =[],[]
for d in te :
    if not len (d ["G"]):
        continue 
    k =maske (d ["gate_skor"],0.40 ,0.30 )
    P ,D =(d ["P"][k ],d ["D"][k ])if k .any ()else (d ["P"][:0 ],d ["D"][:0 ])
    f =f"{OB }/{d ['pid']}.npz"
    if len (P )and os .path .exists (f ):
        z =np .load (f )
        P ,D =product_chain .tam_poz (
        np .ascontiguousarray (z ["V"],np .float64 ),
        np .ascontiguousarray (z ["F"],np .int64 ),
        np .asarray (z ["pbs"],float ).mean (0 ),P ,D ,step_path =S .get (d ["pid"]))
    T .append ((len (d ["G"]),)+match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],0. ,180. ,True )[:3 ])
    R .append ((len (d ["G"]),)+match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],YANAL ,ACI ,
    False ,signed =True )[:3 ])
tA ,rA =f1w (T ),f1w (R )
print (f"\nA) MANSET (gate -> tam_poz)        detection {tA :.4f} | robot {rA :.4f}",flush =True )

# B) p5-v2 KOLU
T ,R =[],[]
for d in te :
    if not len (d ["G"]):
        continue 
    score =[clf .predict_proba (np .asarray ([s [2 ]for s in o ],float ))[:,1 ]
    for o in d ["secs"]]
    P ,D =PE .sec (d ["secs"],score ,gate_skor =d ["gate_skor"],gate_esik =(0.40 ,0.30 ))
    T .append ((len (d ["G"]),)+match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],0. ,180. ,True )[:3 ])
    R .append ((len (d ["G"]),)+match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],YANAL ,ACI ,
    False ,signed =True )[:3 ])
tB ,rB =f1w (T ),f1w (R )
print (f"B) p5-v2 (ortak secim -> gate)     detection {tB :.4f} | robot {rB :.4f}")
print (f"\nFARK: detection {tB -tA :+.4f} | robot {rB -rA :+.4f} (goreli %{100 *(rB /max (rA ,1e-9 )-1 ):.0f})")
json .dump ({"damga":receipt_hash .damga (),"headline":{"detection":tA ,"robot":rA },
"p5v2":{"detection":tB ,"robot":rB },"n_parca":len (te ),
"not":"D7 = DEV, FINAL DEGIL. p5-v2 D6'da egitildi (brand-ayrik)."},
open ("results/d7_headline_p5v2.json","w"),indent =1 )
print ("receipt -> results/d7_headline_p5v2.json")
