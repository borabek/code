# -*- coding: utf-8 -*-
"""HIBRIT + AGIZ TANIMLAYICILARI: B-rep modeline YENI BILGI vererek yeniden dene.

Onceki hibritte B-rep adaylari only 58 gate ozniteligiyle puanlaniyordu and arm
NULL output (ceiling 0.5748 but uctan uca 0.1959 <= baseline 0.1970). Olculdu ki that 58
feature a agzin real tel girisi olup olmadigini SOYLEMIYOR.

`mouth_descriptor.py` that bilgiyi uretti. Parca-ICI AUC (training absent, saf ayrilabilirlik,
`results/agiz_ayirt_edicilik.json`): radius 0.690, es_eksen 0.293, narinlik 0.317,
girme_kenar 0.331, girme 0.343 -- 0.29 with 0.69 equal guclu, direction ters.

MIMARI DEGISMEDI and this KASITLI: seg adaylari DAGITILAN v6 with puanlanir (tez-saf
arm AYNEN korunur, arm kotu calisirsa baseline KAYBEDILMEZ); only B-rep modeli
58 + 9 = 67 sutunla, part-ici z-skor donusumuyle yeniden egitilir.

TEZE SADIK: segmentasyon, ~6000 remesh, `v_o` turetmesi DEGISMEDI.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 
from sklearn .ensemble import RandomForestClassifier 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import mouth_descriptor as AT # noqa: E402
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402
import wire_gate # noqa: E402
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OZ ="results/_brep_oz"
TAN ="results/_agiz_tan"
DONUSUM ="zskor"


def tanim (on ,pid ,n_bek ):
    """Agiz tanimlayicilari. Sayi tutmazsa SESSIZ KABUL YOK -- None returns."""
    f =f"{TAN }/{on }_{pid }.npz"
    if not os .path .exists (f ):
        return None 
    T =np .asarray (np .load (f )["T"],float )
    return T if len (T )==n_bek else None 


def egitim_seti ():
    R ={str (r ["pid"]):r for r in pickle .load (open (K .KAYIT ,"rb"))}
    R6 =d6_record .yukle (set (d6_record .exam ()["pidler"]))
    M ,Y =[],[]
    uyumsuz =0 
    for f in sorted (os .listdir (OZ )):
        if not (f .endswith (".npz")and (f .startswith ("tam_")or f .startswith ("d6_"))):
            continue 
        on ="tam"if f .startswith ("tam_")else "d6"
        pid =f [len (on )+1 :-4 ]
        r =R .get (pid )or R6 .get (pid )
        if r is None :
            continue 
        z =np .load (f"{OZ }/{f }")
        X =np .asarray (z ["X"],float )
        nseg =len (np .asarray (r ["P"],float ))
        if nseg >len (X ):
            continue 
        Xb ,yb =X [nseg :],z ["y"][nseg :]
        if len (Xb )<2 :
            continue 
        T =tanim (on ,pid ,len (Xb ))
        if T is None :
            uyumsuz +=1 
            continue 
        M .append (wire_gate .within_part (np .hstack ([Xb ,T ]),DONUSUM ))
        Y .append (yb )
    print (f"  tanimlayici sayisi tutmayan/eksik part: {uyumsuz }",flush =True )
    return np .vstack (M ),np .concatenate (Y )


def sinav_seti ():
    te =[]
    atlanan =0 
    for f in sorted (os .listdir (OZ )):
        if not (f .startswith ("d7_")and f .endswith (".npz")):
            continue 
        pid =f [3 :-4 ]
        z =np .load (f"{OZ }/{f }")
        X =np .asarray (z ["X"],float )
        kay =z ["source"]
        nb =int ((kay ==1 ).sum ())
        T =tanim ("d7",pid ,nb )if nb else np .zeros ((0 ,len (AT .AD )))
        if T is None :
            atlanan +=1 
            continue 
        te .append ({"pid":pid ,"X":X ,"T":T ,"P":z ["P"],"D":z ["D"],
        "source":kay })
    print (f"  tanimlayicisi eksik exam parcasi: {atlanan }",flush =True )
    return te 


def main ():
    print ("training seti kuruluyor...",flush =True )
    M ,Y =egitim_seti ()
    print (f"B-rep modeli: {M .shape } | pozitif {Y .mean ():.4f}",flush =True )
    bclf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M ,Y )
    bmod ={"clf":bclf ,"cols":None ,"n_feat":M .shape [1 ],"donusum":DONUSUM }
    pickle .dump (bmod ,open ("results/brep_gate_tanimlayicili.pkl","wb"))

    te =sinav_seti ()
    kay =K .yukle ([d ["pid"]for d in te ])
    for d in te :
        r =kay [d ["pid"]]
        d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
        "Gd":np .asarray (r ["Gd"],float ),"diag":r ["diag"]})
    g6 =K .gate_yukle ()
    print (f"SINAV {len (te )} part\n",flush =True )

    def kos (b_esik ):
        en =None 
        for tip ,e in ([("mutlak",x )for x in (0.30 ,0.40 )]+
        [("goreli",x )for x in ((0.5 ,0.20 ),(0.5 ,0.30 ))]):
            rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
            tes =[]
            for d in te :
                ms =d ["source"]==0 
                Xs =d ["X"][ms ]
                if len (Xs )<2 :
                    continue 
                ss =np .asarray (wire_gate .decision_score (g6 ,Xs ),float )
                ks =(ss >=e )if tip =="mutlak"else maske (ss ,e [0 ],e [1 ])
                P =list (d ["P"][ms ][ks ]);D =list (d ["D"][ms ][ks ]);S =list (ss [ks ])
                if b_esik is not None and len (d ["T"])>=2 :
                    Xb =np .hstack ([d ["X"][~ms ],d ["T"]])
                    sb =np .asarray (wire_gate .decision_score (bmod ,Xb ),float )
                    kb =sb >=b_esik 
                    P +=list (d ["P"][~ms ][kb ]);D +=list (d ["D"][~ms ][kb ])
                    S +=list (sb [kb ])
                P =np .asarray (P ,float ).reshape (-1 ,3 )
                D =np .asarray (D ,float ).reshape (-1 ,3 )
                if len (P )>1 :
                    nm =wire_gate .crowd_mask (P ,np .asarray (S ,float ))
                    P ,D =P [nm ],D [nm ]
                tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
                K .ACI ,False ,signed =True )[:3 ]
                a =rob [d ["mfg"]]
                a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
                tes .append ((len (d ["G"]),)+match_hungarian (
                P ,D ,d ["G"],d ["Gd"],d ["diag"],
                max (3.0 ,0.06 *d ["diag"]),180.0 ,True )[:3 ])
            pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )
            for m ,a in rob .items ()}
            mi =float (2 *sum (a [0 ]for a in rob .values ())/
            max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
            r ={"rule":f"{tip } {e }","robot":mi ,"tespit":K .mikro (tes ),
            "makro":float (np .mean (list (pm .values ()))),
            "en_kotu":float (min (pm .values ())),"brand":pm }
            if en is None or r ["robot"]>en ["robot"]:
                en =r 
        return en 

    out ={}
    for ad ,be in (("TEZ-SAF (B-rep KAPALI)",None ),("+B-rep 0.40",0.40 ),
    ("+B-rep 0.50",0.50 ),("+B-rep 0.60",0.60 ),
    ("+B-rep 0.70",0.70 ),("+B-rep 0.80",0.80 )):
        out [ad ]=kos (be )
        r =out [ad ]
        print (f"{ad :<24} robot {r ['robot']:.4f} | tespit {r ['tespit']:.4f} | "
        f"makro {r ['makro']:.4f} | en kotu {r ['en_kotu']:.4f} | {r ['rule']}",
        flush =True )
    t =out ["TEZ-SAF (B-rep KAPALI)"]
    print ()
    for ad in out :
        if ad =="TEZ-SAF (B-rep KAPALI)":
            continue 
        r =out [ad ]
        art =sum (1 for m in r ["brand"]if r ["brand"][m ]>t ["brand"][m ]+1e-9 )
        yik =[m for m in r ["brand"]if r ["brand"][m ]==0 and t ["brand"][m ]>0 ]
        print (f"  {ad :<14} robot {r ['robot']-t ['robot']:+.4f} | tespit "
        f"{r ['tespit']-t ['tespit']:+.4f} | artan brand {art }/12"
        +(f" | YIKILAN: {','.join (yik )}"if yik else ""))
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,
    "not":"HIBRIT + mouth tanimlayicilari (58+9=67 sutun, part-ici "
    "zskor). Seg kolu DAGITILAN v6, degistirilmedi. D7 "
    "brand-disi, MIKRO. B-rep TEZ TURETMESI DEGIL."},
    open ("results/hibrit_tanimlayicili_d7.json","w"),indent =1 )
    print ("receipt -> results/hibrit_tanimlayicili_d7.json")


if __name__ =="__main__":
    main ()
