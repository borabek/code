# -*- coding: utf-8 -*-
"""U2: topoloji YARICAPI (R) UCTAN UCA -- tanidik + each two gorulmemis manufacturer.

T19 candidate duzeyinde R=12'yi +0.0100 with onde buldu (R=6 -> 8 -> 12 tutarli artis). Aday duzeyi
this gece UC KEZ yaniltti, that yuzden karar BURADA veriliyor.

TASARIM (single degisken R olsun diye):
  * candidates BIR KEZ turetilir (network + oy birlesimi + router) -- R'den bagimsiz
  * 18 baseline+fiziksel column BIR KEZ is computed -- R'den bagimsiz
  * icbukey KENAR YAPISI part basina BIR KEZ -- R'den bagimsiz
  * only 4 topoloji sutunu each R for yeniden degerlendirilir
  * each R own training npz'siyle eslesir (u1_topo_r_yeniden.py with uretildi, R=6'da BIREBIR
    dogrulandi) -- EGITIM and TEST AYNI R'de olmak ZORUNDA

KILL (onceden yazili): new R, (a) tanidik tespit F1'de R=6'ya according to >= -0.01 kalmali VE
(b) gorulmemis manufacturer EN KOTU durumunu ARTIRMALI. Ikisi birden olmazsa DAGITILMAZ.
(t15'in TOPO for kullandigi killin aynisi -- same arm, same cubuk.)
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

R_LISTE =[6.0 ,8.0 ,12.0 ]
NPZ_OF ={6.0 :"results/gate_regrow_data_topo.npz",
8.0 :"results/gate_regrow_data_topo_r8.npz",
12.0 :"results/gate_regrow_data_topo_r12.npz"}


def main ():
    import cp_openings ,robot_cp ,topo_feats ,wire_gate 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sina_cluster import esle ,f1w ,pr 
    from sklearn .ensemble import RandomForestClassifier 

    for r_ ,f_ in NPZ_OF .items ():
        assert os .path .exists (f_ ),f"R={r_ } icin training verisi YOK: {f_ }"
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
    CL =float (pp ["cluster_mm"])
    ORAN =float (cfg ["gate_goreli_oran"]);TABAN =float (cfg ["gate_goreli_taban"])
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}

    cache =[]
    for k in ("dev","val"):
        cf =f"results/_probs_{k }.pkl"
        if not os .path .exists (cf )and k =="dev":
            cf ="results/_h_probs.pkl"
        for r in pickle .load (open (cf ,"rb")):
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
            r ["cluster"]=k 
            cache .append (r )
    print (f"{len (cache )} part | "
    f"{dict (collections .Counter (mfg_of .get (r ['pid'],'?')for r in cache ))}",flush =True )

    DER =[]
    for i ,r in enumerate (cache ,1 ):
        if i %25 ==0 :
            print (f"  turetme {i }/{len (cache )}",flush =True )
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
        X18 ,XR =None ,{}
        if cps :
            X18 =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,
            step_path =r ["stp"])[:,:18 ]
            onb =topo_feats .icbukey_kenarlar (V ,F )# R'den BAGIMSIZ, part basina 1 times
            P_ =np .array ([c ["point"]for c in cps ],float )
            D_ =np .array ([c ["direction"]for c in cps ],float )
            for R_ in R_LISTE :
                XR [R_ ]=np .array ([topo_feats .topo_ozellik (V ,F ,P_ [k ],D_ [k ],R =R_ ,cache =onb )
                for k in range (len (P_ ))],float )
        DER .append (dict (pid =r ["pid"],cluster =r ["cluster"],mfg =mfg_of .get (r ["pid"],"?"),
        X18 =X18 ,XR =XR ,
        P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))
    print (f"turetme bitti | notr-donus: {wire_gate .fallback_ozet ()or 'YOK'}",flush =True )

    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (r ["pid"],"yok:"+r ["pid"])for r in DER }

    def puanla (clf ,alt ,R_ ):
        det ,rob =[],[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X18"]is not None :
                X =np .hstack ([r ["X18"],r ["XR"][R_ ]])
                s =clf .predict_proba (X )[:,1 ]
                m =(s >=ORAN *max (float (s .max ()),1e-9 ))&(s >=TABAN )
                if m .any ():
                    P =r ["P"][m ];Pd =r ["Pd"][m ]
            k ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    print (f"\n{'split':<24}{'R':>6}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}")
    R ={}
    for R_ in R_LISTE :
        d =np .load (NPZ_OF [R_ ],allow_pickle =True )
        tr_pid =np .array ([str (x )for x in d ["pids"]])
        tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
        tr_mfg =np .array ([str (x )for x in d ["mfg"]])
        kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in tr_pid [tr_mfg ==k ]).most_common (1 )[0 ][0 ]
        for k in np .unique (tr_mfg )}
        kur =lambda keep :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,
        n_jobs =-1 ,random_state =0 ).fit (
        d ["X"][keep ],d ["y"][keep ])
        keep =~np .isin (tr_grp ,list (tg ))
        det ,rob =puanla (kur (keep ),DER ,R_ )
        p_ ,r_ =pr (det )
        print (f"{'TANIDIK (DEV+VAL)':<24}{R_ :>6.1f}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}"
        f"{p_ :>9.3f}{r_ :>9.3f}",flush =True )
        R [(R_ ,"tanidik")]=(det ,rob )
        for k ,mad in kod .items ():
            alt =[x for x in DER if x ["mfg"]==mad ]
            if len (alt )<10 :
                continue 
            det2 ,rob2 =puanla (kur ((tr_mfg !=k )&~np .isin (tr_grp ,list (tg ))),alt ,R_ )
            p2 ,r2 =pr (det2 )
            print (f"{'  '+mad +' disarida ('+str (len (alt ))+')':<24}{R_ :>6.1f}"
            f"{f1w (det2 ):>9.4f}{f1w (rob2 ):>9.4f}{p2 :>9.3f}{r2 :>9.3f}",flush =True )
            R [(R_ ,mad )]=(det2 ,rob2 )
        print ()

    key_ =sorted ({k [1 ]for k in R if k [1 ]!="tanidik"})
    taban_tan =f1w (R [(6.0 ,"tanidik")][0 ])
    taban_kotu =min (f1w (R [(6.0 ,a )][0 ])for a in key_ )
    print (f"KARAR (baseline R=6.0: tanidik {taban_tan :.4f} | gorulmemis en kotu {taban_kotu :.4f})")
    kazanan ,en_iyi =None ,None 
    for R_ in R_LISTE :
        if R_ ==6.0 :
            continue 
        dt =f1w (R [(R_ ,"tanidik")][0 ])-taban_tan 
        ek =min (f1w (R [(R_ ,a )][0 ])for a in key_ )
        gecti =(dt >=-0.01 )and (ek >taban_kotu )
        print (f"  R={R_ :<5} tanidik {dt :+.4f} | en kotu {taban_kotu :.4f} -> {ek :.4f} "
        f"({ek -taban_kotu :+.4f}) -> {'GECTI'if gecti else 'GECMEDI'}")
        if gecti and (en_iyi is None or ek >en_iyi ):
            kazanan ,en_iyi =R_ ,ek 
    print (f"\nSONUC: {'R='+str (kazanan )+' DAGITILABILIR'if kazanan else 'HICBIRI GECMEDI -> R=6.0 KALIR'}")
    json .dump ({f"{k [0 ]}|{k [1 ]}":{"tespit":float (f1w (v [0 ])),"robot":float (f1w (v [1 ]))}
    for k ,v in R .items ()}|{"kazanan":kazanan },
    open ("results/u2_topo_r_uctan_uca.json","w"),indent =1 )
    pickle .dump (R ,open ("results/u2_parca.pkl","wb"))
    print ("receipt -> results/u2_topo_r_uctan_uca.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ()
        raise 
