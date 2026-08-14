# -*- coding: utf-8 -*-
"""GENISLETILMIS HAVUZ + URUNUN KENDI GATE RECETESI.

ONCEKI KOSUNUN KUSURU: gate'i elle `RandomForestClassifier(...).fit(X, y)` diye
kurdum. Urunun gate'i oyle does not work -- `wire_gate.decision_score` before
`within_part(X, donusum)` uyguluyor (58 -> 116 column, PARCA PARCA z-score) after
`_cokus_yonlendir` with yonlendiriyor. Parca-ici z-score this projenin most large single
gate kazancidir (+0.1273, `gate-refit-minv4`). Onsuz egitilen gate, urunun
gate'i DEGILDIR and kiyas gecersizdir.

Burada `refit_gate_v7.py` recetesi birebir is used (within_part PARCA PARCA,
RF 400/leaf3, urun model sozlugu) and measurement `wire_gate.decision_score` uzerinden
is done -- i.e. measurement yolu = urun yolu.

Ayrica IKI HAVUZ same recete with egitilir ki kiyas TEK DEGISKENLI olsun:
  A) TEZ-SAF pool  (only segmentasyon `v_o` adaylari)
  B) GENISLETILMIS  (seg + B-rep fiziksel onerileri)
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import wire_gate ,canonical_d7 as K 
from sklearn .ensemble import RandomForestClassifier 
from p1c_threshold import maske 
from sina_cluster import match_hungarian 

OZ ="results/_brep_oz"
DONUSUM ="zskor"


def parts (on ,kaynakli ):
    """ten: 'full'|'d6'|'d7'. kaynakli=True whereas `source` alani da returns."""
    v =[]
    for f in sorted (os .listdir (OZ )):
        if not (f .startswith (on +"_")and f .endswith (".npz")):
            continue 
        z =np .load (f"{OZ }/{f }")
        d ={"pid":f [len (on )+1 :-4 ],"X":z ["X"],"y":z ["y"]}
        if kaynakli :
            if "source"not in z :
                continue 
            d .update ({"P":z ["P"],"D":z ["D"],"source":z ["source"]})
        v .append (d )
    return v 


tr =parts ("tam",False )+parts ("d6",False )
te =parts ("d7",True )
kay7 =K .yukle ([d ["pid"]for d in te ])
for d in te :
    r =kay7 [d ["pid"]]
    d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
    "Gd":np .asarray (r ["Gd"],float ),"diag":r ["diag"]})
print (f"EGITIM {len (tr )} part | SINAV {len (te )} part",flush =True )
# D6 parcalarinda source present; egitimde de tez-saf/genisletilmis ayrimi gerekli
d6k ={d ["pid"]:d for d in parts ("d6",True )}
tam_k ={}
for f in sorted (os .listdir (OZ )):
    if f .startswith ("tam_")and f .endswith (".npz"):
        tam_k [f [4 :-4 ]]=None # tam_* dosyalarinda source SAKLANMADI


def egit (segtek ):
    """Urunun recetesi: within_part PARCA PARCA, RF 400/leaf3, urun model sozlugu."""
    M ,Y =[],[]
    skipped =0 
    for d in tr :
        X ,y =d ["X"],d ["y"]
        if segtek :
            k =d6k .get (d ["pid"])
            if k is None :
                skipped +=1 # tam_* for source absent -> tez-saf arm D6 with sinirli
                continue 
            m =k ["source"]==0 
            X ,y =k ["X"][m ],k ["y"][m ]
        if not len (X ):
            continue 
        M .append (wire_gate .within_part (np .asarray (X ,float ),DONUSUM ))
        Y .append (y )
    M =np .vstack (M );Y =np .concatenate (Y )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M ,Y )
    return {"clf":clf ,"cols":None ,"n_feat":M .shape [1 ],"donusum":DONUSUM ,
    "feat_names":None },M .shape ,float (Y .mean ()),skipped 


def olc (model ,segtek ):
    en =None 
    for tip ,e in ([("mutlak",x )for x in (0.20 ,0.30 ,0.40 ,0.50 )]+
    [("goreli",x )for x in ((0.5 ,0.20 ),(0.5 ,0.30 ),(0.4 ,0.25 ))]):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ]);tes =[]
        for d in te :
            m0 =d ["source"]==0 if segtek else np .ones (len (d ["y"]),bool )
            X =np .asarray (d ["X"][m0 ],float )
            if not len (X ):
                continue 
            s =np .asarray (wire_gate .decision_score (model ,X ),float )
            k =(s >=e )if tip =="mutlak"else maske (s ,e [0 ],e [1 ])
            P ,D =d ["P"][m0 ],d ["D"][m0 ]
            P ,D =(P [k ],D [k ])if k .any ()else (P [:0 ],D [:0 ])
            if len (P )>1 :
                nm =wire_gate .crowd_mask (P ,s [k ]);P ,D =P [nm ],D [nm ]
            tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
            K .ACI ,False ,signed =True )[:3 ]
            a =rob [d ["mfg"]];a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
            tes .append ((len (d ["G"]),)+match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
            max (3.0 ,0.06 *d ["diag"]),180.0 ,True )[:3 ])
        pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
        mi =float (2 *sum (a [0 ]for a in rob .values ())/
        max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
        r ={"rule":f"{tip } {e }","robot":mi ,"detection":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ())),"brand":pm }
        print (f"    {r ['rule']:<16} robot {mi :.4f} | detection {r ['detection']:.4f} | "
        f"makro {r ['makro']:.4f}",flush =True )
        if en is None or r ["robot"]>en ["robot"]:
            en =r 
    return en 


out ={}
for ad ,segtek in (("GENISLETILMIS (seg + B-rep)",False ),):
    print (f"\n{ad }",flush =True )
    model ,sh ,poz ,atl =egit (segtek )
    print (f"  training {sh } | pozitif {poz :.4f}",flush =True )
    pickle .dump (model ,open ("results/brep_gate_recete.pkl","wb"))
    out [ad ]=olc (model ,segtek )
    print (f"  EN IYI: {out [ad ]['rule']} robot {out [ad ]['robot']:.4f}",flush =True )

g =out ["GENISLETILMIS (seg + B-rep)"]
print (f"\nKANONIK TABAN (tez-saf, dagitilan gate v6 + NMS): robot 0.2029 / detection 0.4523")
print (f"GENISLETILMIS (urun recetesi):                     robot {g ['robot']:.4f} / "
f"detection {g ['detection']:.4f}  ({g ['robot']-0.2029 :+.4f})")
print (f"Genisletilmis pool tavani 0.5748 -> yakalanan pay %{100 *g ['robot']/0.5748 :.1f}")
json .dump ({"damga":receipt_hash .damga (),"sonuc":out ,
"taban_tez_saf":{"robot":0.2029 ,"detection":0.4523 },
"tavan_genisletilmis":0.5748 ,
"not":"Urunun gate recetesi (within_part zskor PARCA PARCA, RF 400/leaf3) "
"ve urunun decision_score with measured. D7 brand-disi, MIKRO. "
"B-rep TEZ TURETMESI DEGIL, ek candidate kaynagi."},
open ("results/brep_gate_recipe_d7.json","w"),indent =1 )
print ("receipt -> results/brep_gate_recipe_d7.json")
