# -*- coding: utf-8 -*-
"""HIBRIT GATE: v6 seg adaylarini, AYRI a model B-rep adaylarini puanlar.

WHY BU MIMARI:
* Dagitilan v6 tez-saf havuzda 0.1970 veriyor; benim same recete with elde
  kurdugum gate 0.1159. Yani v6'yi YENIDEN kurmaya calismak ~0.08 kaybettiriyor.
* v6'yi genisletilmis havuza oldugu like vermek de dusuruyor (0.1435): B-rep
  adaylarini HIC gormedi, onlari wrong puanliyor ([[gate-refit-minv4]]).
* Cozum: v6'ya DOKUNMA. Seg adaylari AYNEN v6 with puanlanir -- i.e. dagitilan
  baseline birebir korunur. B-rep adaylari AYRI a model with puanlanir and however
  KENDI esigini gecerse havuza katilir. Kol kotu calisirsa baseline KAYBEDILMEZ.

TEZE SADIKLIK: `v_o` turetmesi, 5 sinif, ~6000 remesh DEGISMEDI. Seg kolu
tezin kendisidir and dokunulmamistir; B-rep kolu EK a candidate kaynagidir and
raporlarda AYRI satirda gosterilir.
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

# --- B-rep modeli: YALNIZ B-rep adaylariyla egitilir ------------------------
M ,Y =[],[]
for f in sorted (os .listdir (OZ )):
    if not (f .endswith (".npz")and (f .startswith ("tam_")or f .startswith ("d6_"))):
        continue 
    on ="tam_"if f .startswith ("tam_")else "d6_"
    pid =f [len (on ):-4 ]
    r =R .get (pid )or R6 .get (pid )
    if r is None :
        continue 
    z =np .load (f"{OZ }/{f }");X =np .asarray (z ["X"],float );y =z ["y"]
    nseg =len (np .asarray (r ["P"],float ))
    if nseg >len (X ):
        continue 
    Xb ,yb =X [nseg :],y [nseg :]
    if len (Xb )<2 :
        continue 
    M .append (wire_gate .within_part (Xb ,DONUSUM ));Y .append (yb )
M =np .vstack (M );Y =np .concatenate (Y )
print (f"B-rep modeli egitimi: {M .shape } | pozitif {Y .mean ():.4f}",flush =True )
bclf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
random_state =0 ).fit (M ,Y )
bmod ={"clf":bclf ,"cols":None ,"n_feat":M .shape [1 ],"donusum":DONUSUM }
pickle .dump (bmod ,open ("results/brep_only_gate.pkl","wb"))

# --- Sinav ------------------------------------------------------------------
te =[]
for f in sorted (os .listdir (OZ )):
    if f .startswith ("d7_")and f .endswith (".npz"):
        z =np .load (f"{OZ }/{f }")
        te .append ({"pid":f [3 :-4 ],"X":np .asarray (z ["X"],float ),"P":z ["P"],
        "D":z ["D"],"source":z ["source"]})
kay =K .yukle ([d ["pid"]for d in te ])
for d in te :
    r =kay [d ["pid"]]
    d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
    "Gd":np .asarray (r ["Gd"],float ),"diag":r ["diag"]})
g6 =K .gate_yukle ()
print (f"SINAV {len (te )} part",flush =True )


def kos (b_esik ):
    """b_esik=None -> B-rep kolu KAPALI (tez-saf baseline)."""
    en =None 
    for tip ,e in ([("mutlak",x )for x in (0.30 ,0.40 )]+
    [("goreli",x )for x in ((0.5 ,0.20 ),(0.5 ,0.30 ),(0.4 ,0.25 ))]):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ]);tes =[]
        for d in te :
            ms =d ["source"]==0 
            Xs =d ["X"][ms ]
            if len (Xs )<2 :
                continue 
            ss =np .asarray (wire_gate .decision_score (g6 ,Xs ),float )
            ks =(ss >=e )if tip =="mutlak"else maske (ss ,e [0 ],e [1 ])
            P =list (d ["P"][ms ][ks ]);D =list (d ["D"][ms ][ks ]);S =list (ss [ks ])
            if b_esik is not None :
                mb =~ms 
                Xb =d ["X"][mb ]
                if len (Xb )>=2 :
                    sb =np .asarray (wire_gate .decision_score (bmod ,Xb ),float )
                    kb =sb >=b_esik 
                    P +=list (d ["P"][mb ][kb ]);D +=list (d ["D"][mb ][kb ])
                    S +=list (sb [kb ])
            P =np .asarray (P ,float ).reshape (-1 ,3 )
            D =np .asarray (D ,float ).reshape (-1 ,3 )
            if len (P )>1 :
                nm =wire_gate .crowd_mask (P ,np .asarray (S ,float ))
                P ,D =P [nm ],D [nm ]
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
        "en_kotu":float (min (pm .values ())),"brand":pm }
        if en is None or r ["robot"]>en ["robot"]:
            en =r 
    return en 


out ={}
for ad ,be in (("TEZ-SAF (B-rep KAPALI)",None ),("+B-rep threshold 0.50",0.50 ),
("+B-rep threshold 0.60",0.60 ),("+B-rep threshold 0.70",0.70 ),
("+B-rep threshold 0.80",0.80 ),("+B-rep threshold 0.90",0.90 )):
    out [ad ]=kos (be )
    r =out [ad ]
    print (f"{ad :<24} robot {r ['robot']:.4f} | tespit {r ['tespit']:.4f} | makro "
    f"{r ['makro']:.4f} | en kotu {r ['en_kotu']:.4f} | {r ['rule']}",flush =True )
t =out ["TEZ-SAF (B-rep KAPALI)"]
print ()
for ad in out :
    if ad !="TEZ-SAF (B-rep KAPALI)":
        r =out [ad ]
        art =sum (1 for m in r ["brand"]if r ["brand"][m ]>t ["brand"][m ]+1e-9 )
        yik =[m for m in r ["brand"]if r ["brand"][m ]==0 and t ["brand"][m ]>0 ]
        print (f"  {ad :<20} robot {r ['robot']-t ['robot']:+.4f} | tespit "
        f"{r ['tespit']-t ['tespit']:+.4f} | artan brand {art }/12"
        +(f" | YIKILAN: {','.join (yik )}"if yik else ""))
json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,
"not":"HIBRIT: seg adaylari DAGITILAN v6 with, B-rep adaylari AYRI "
"model with puanlanir. Tez-saf arm DEGISMEDI. D7 brand-disi, MIKRO."},
open ("results/hibrit_gate_d7.json","w"),indent =1 )
print ("receipt -> results/hibrit_gate_d7.json")
