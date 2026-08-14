# -*- coding: utf-8 -*-
"""TEK DEGISKENLI KIYAS: same recete, same parts, TEZ-SAF vs GENISLETILMIS pool.

Genisletilmis pool uctan uca 0.1515 verdi, kanonik baseline 0.2029. Ama kanonik
baseline DAGITILAN gate v6 (baska corpus, baska recete, `_cokus_yonlendir` dahil);
benimki elde kuruldu. Bu farki havuza yazmak HATA becomes.

Burada single degisken HAVUZ: same training parcalari, same recete (within_part zskor
PARCA PARCA, RF 400/leaf3), same threshold taramasi, same NMS.

KAYNAK ISARETI YENIDEN HESAP GEREKTIRMIYOR: `brep_pool.merged_pool`
segmentasyon adaylarini HER ZAMAN basa koyuyor, i.e. first `len(r["P"])` candidate
seg, gerisi B-rep. Kayittan birebir kurulur.
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import wire_gate ,canonical_d7 as K ,d6_record 
from sklearn .ensemble import RandomForestClassifier 
from p1c_threshold import maske 
from sina_cluster import match_hungarian 

OZ ="results/_brep_oz";DONUSUM ="zskor"
R ={str (r ["pid"]):r for r in pickle .load (open (K .KAYIT ,"rb"))}
R6 =d6_record .yukle (set (d6_record .exam ()["pidler"]))

tr =[]
for f in sorted (os .listdir (OZ )):
    if f .endswith (".npz")and (f .startswith ("tam_")or f .startswith ("d6_")):
        on ="tam_"if f .startswith ("tam_")else "d6_"
        pid =f [len (on ):-4 ]
        r =R .get (pid )or R6 .get (pid )
        if r is None :
            continue 
        z =np .load (f"{OZ }/{f }")
        nseg =len (np .asarray (r ["P"],float ))
        X =z ["X"]
        if nseg >len (X ):
            continue # tutarsiz cache -> atla (silent kabul YOK)
        kay =np .zeros (len (X ),int );kay [nseg :]=1 
        tr .append ({"pid":pid ,"X":X ,"y":z ["y"],"source":kay })
te =[]
for f in sorted (os .listdir (OZ )):
    if f .startswith ("d7_")and f .endswith (".npz"):
        z =np .load (f"{OZ }/{f }")
        te .append ({"pid":f [3 :-4 ],"X":z ["X"],"y":z ["y"],"P":z ["P"],
        "D":z ["D"],"source":z ["source"]})
kay7 =K .yukle ([d ["pid"]for d in te ])
for d in te :
    r =kay7 [d ["pid"]]
    d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
    "Gd":np .asarray (r ["Gd"],float ),"diag":r ["diag"]})
segp =sum (int ((d ["source"]==0 ).sum ())for d in tr )
print (f"EGITIM {len (tr )} part | seg candidate {segp } | toplam candidate "
f"{sum (len (d ['X'])for d in tr )} | SINAV {len (te )}",flush =True )


def egit (segtek ):
    M ,Y =[],[]
    for d in tr :
        m =(d ["source"]==0 )if segtek else np .ones (len (d ["X"]),bool )
        X ,y =np .asarray (d ["X"][m ],float ),d ["y"][m ]
        if len (X )<2 :
            continue 
        M .append (wire_gate .within_part (X ,DONUSUM ));Y .append (y )
    M =np .vstack (M );Y =np .concatenate (Y )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M ,Y )
    return ({"clf":clf ,"cols":None ,"n_feat":M .shape [1 ],"donusum":DONUSUM },
    M .shape ,float (Y .mean ()))


def olc (model ,segtek ):
    en =None 
    for tip ,e in ([("mutlak",x )for x in (0.20 ,0.30 ,0.40 )]+
    [("goreli",x )for x in ((0.5 ,0.20 ),(0.5 ,0.30 ))]):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ]);tes =[]
        for d in te :
            m0 =(d ["source"]==0 )if segtek else np .ones (len (d ["y"]),bool )
            X =np .asarray (d ["X"][m0 ],float )
            if len (X )<2 :
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
        r ={"rule":f"{tip } {e }","robot":mi ,"tespit":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ()))}
        if en is None or r ["robot"]>en ["robot"]:
            en =r 
    return en 


out ={}
for ad ,segtek in (("TEZ-SAF pool",True ),("GENISLETILMIS pool",False )):
    model ,sh ,poz =egit (segtek )
    out [ad ]=olc (model ,segtek )
    r =out [ad ]
    print (f"{ad :<22} robot {r ['robot']:.4f} | tespit {r ['tespit']:.4f} | makro "
    f"{r ['makro']:.4f} | en kotu {r ['en_kotu']:.4f} | {r ['rule']} | "
    f"training {sh } poz {poz :.4f}",flush =True )
a ,b =out ["TEZ-SAF pool"],out ["GENISLETILMIS pool"]
print (f"\nHAVUZUN TEK-DEGISKENLI ETKISI: robot {b ['robot']-a ['robot']:+.4f} | "
f"tespit {b ['tespit']-a ['tespit']:+.4f} | makro {b ['makro']-a ['makro']:+.4f}")
print (f"(dagitilan gate v6 + NMS, tez-saf pool: robot 0.2029 / tespit 0.4523 -- "
f"AYRI RECETE, kiyas icin degil referans)")
json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,
"havuz_etkisi_robot":b ["robot"]-a ["robot"],
"havuz_etkisi_tespit":b ["tespit"]-a ["tespit"],
"not":"TEK DEGISKEN = pool. Ayni parts, same recete, same threshold "
"taramasi, same NMS. D7 brand-disi, MIKRO."},
open ("results/havuz_tek_degisken.json","w"),indent =1 )
