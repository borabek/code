# -*- coding: utf-8 -*-
"""T7: GORELI ESIK uctan uca -- tanidik veride ne kaybettiriyor?

ADAY DUZEYINDE MEASURED (t6): goreli 0.5 + baseline 0.20
  tanidik (geometri-disi) 0.7422 -> 0.7348 (-0.0074)
  manufacturer-disi EN KOTU    0.2799 -> 0.4402 (+0.160)

AMA candidate-duzeyi F1 uctan uca F1 DEGILDIR (E maddesi gate duzeyinde kazanip uctan uca kaybetmisti).
Bu betik same candidates and same skorlar on YALNIZ DECISION KURALINI changes -- single degisken,
yeniden inference absent, yeniden turetme absent.

KILL (onceden yazili): tanidik (DEV+VAL) tespit F1 kaybi 0.02'yi asarsa ACILMAZ. Kazanc already
tanidik veride not, GORULMEMIS URETICIDE bekleniyor; here olculen sey BEDEL.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from sina_cluster import esle ,f1w ,pr 

    cluster =(sys .argv [1 ]if len (sys .argv )>1 else "dev").lower ()
    cf =f"results/_probs_{cluster }.pkl"
    if not os .path .exists (cf )and cluster =="dev":
        cf ="results/_h_probs.pkl"
    cache =pickle .load (open (cf ,"rb"))
    for r in cache :
        r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"dusuk":float (cfg ["robot_wire_gate_threshold"]),
    "cok":float (cfg ["robot_wire_gate_threshold_highcp"])}

    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (r ["pid"],"yok:"+r ["pid"])for r in cache }
    pids =np .array ([str (x )for x in d ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"yok:"+p )for p in pids ]),list (tg ))
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (d ["X"][keep ][:,:18 ],d ["y"][keep ])
    print (f"KUME={cluster } | {len (cache )} part | gate {int (keep .sum ())} candidate",flush =True )

    # --- TURETME + SKOR a times; karar kurallari after bedava ---
    DER =[]
    for r in cache :
        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
        mk =lambda pr_ ,**kw :cp_openings .connection_points (
        V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r ["stp"],**kw )
        merge =lambda L :robot_cp ._vote2 (L ,min_votes =1 )
        base =merge ([mk (pb )for pb in plist ])
        is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
        cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
        s =np .zeros (0 )
        if cps :
            Xc =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,step_path =r ["stp"])
            s =clf .predict_proba (Xc [:,:18 ])[:,1 ]
        DER .append (dict (s =s ,is_hi =is_hi ,
        P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))

    def kos (kural ):
        det ,rob =[],[]
        for r in DER :
            s =r ["s"]
            if len (s )==0 :
                m =np .zeros (0 ,bool )
            elif kural =="sabit":
                m =s >=(THR ["cok"]if r ["is_hi"]else THR ["dusuk"])
            else :
                ratio ,baseline =kural 
                m =(s >=ratio *max (float (s .max ()),1e-9 ))&(s >=baseline )
            P =r ["P"][m ]if m .any ()else np .zeros ((0 ,3 ))
            Pd =r ["Pd"][m ]if m .any ()else np .zeros ((0 ,3 ))
            k ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    ADAY =[("sabit (mevcut)","sabit"),("goreli 0.5 + baseline 0.20",(0.5 ,0.20 )),
    ("goreli 0.5 + baseline 0.25",(0.5 ,0.25 )),("goreli 0.4 + baseline 0.20",(0.4 ,0.20 ))]
    print (f"\n{'karar kurali':<26}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}{'fark':>9}")
    R ,taban_f ={},None 
    for ad ,k in ADAY :
        det ,rob =kos (k )
        R [ad ]=(det ,rob )
        if taban_f is None :
            taban_f =f1w (det )
        p_ ,r_ =pr (det )
        print (f"{ad :<26}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}{p_ :>9.3f}{r_ :>9.3f}"
        f"{f1w (det )-taban_f :>+9.4f}",flush =True )
    pickle .dump (R ,open (f"results/t7_parca_{cluster }.pkl","wb"))
    json .dump ({k :{"tespit":float (f1w (v [0 ])),"robot":float (f1w (v [1 ]))}for k ,v in R .items ()},
    open (f"results/t7_goreli_{cluster }.json","w"),indent =1 )
    print (f"\nKILL: tanidik tespit kaybi > 0.02 ise ACILMAZ. receipt -> results/t7_goreli_{cluster }.json")


if __name__ =="__main__":
    main ()
