# -*- coding: utf-8 -*-
"""U4: PARCA-ICI feature normalizasyonu UCTAN UCA (tanidik + two gorulmemis manufacturer).

U3 candidate duzeyinde: most kotu manufacturer-disi 0.4597 -> 0.5219 (+0.062), tanidik bedeli +0.0003.
Aday duzeyi this gece UC KEZ yaniltti (EK blogu, goreli esigin bedeli, sinif dengeleme) --
karar BURADA.

DAGITILABILIRLIK KONTROLU (onemli): donusum PARCA ICINDE tanimli, i.e. calisma aninda a
parcanin KENDI adaylarindan is computed. Korpus istatistigi, komsu part, manufacturer kimligi
GEREKMEZ -- real robot akisinda birebir uygulanabilir. (Aksi a donusum olculebilir but
dagitilamazdi; this ayrimi bilerek yaziyorum.)

KILL (onceden yazili, BUYUKLUK dahil):
  (a) tanidik tespit F1 >= -0.01  VE
  (b) gorulmemis manufacturer EN KOTU durumu >= +0.01 artmali.
Ikisi birden olmazsa DAGITILMAZ. (t15/TOPO with same cubuk + buyukluk sarti.)
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
NJOB =4 # u1 arka planda -- tum cekirdekleri kapma


def _sira (sub ):
    if len (sub )==1 :
        return np .full_like (sub ,0.5 ,dtype =float )
    return np .argsort (np .argsort (sub ,axis =0 ),axis =0 ).astype (float )/(len (sub )-1 )


def _z (sub ):
    sd =sub .std (0 )
    return np .where (sd >1e-12 ,(sub -sub .mean (0 ))/np .where (sd >1e-12 ,sd ,1.0 ),0.0 )


DONUSUM ={"A ham":None ,"B ham+sira":_sira ,"D ham+zskor":_z }


def uygula (X ,fn ):
    """Tek parcanin candidate matrisine donusumu uygula (calisma anindaki islemin AYNISI)."""
    return X if fn is None else np .hstack ([X ,fn (X )])


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sina_cluster import esle ,f1w ,pr 
    from sklearn .ensemble import RandomForestClassifier 

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
    print (f"{len (cache )} part",flush =True )

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
        X =(wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,
        step_path =r ["stp"])if cps else None )
        DER .append (dict (pid =r ["pid"],mfg =mfg_of .get (r ["pid"],"?"),X =X ,
        P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))
    print (f"turetme bitti | notr-donus: {wire_gate .fallback_ozet ()or 'YOK'}",flush =True )

    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    tr_mfg =np .array ([str (x )for x in d ["mfg"]])
    Xtr_ham =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in tr_pid [tr_mfg ==k ]).most_common (1 )[0 ][0 ]
    for k in np .unique (tr_mfg )}
    tg ={gk .get (r ["pid"],"yok:"+r ["pid"])for r in DER }

    XTR ={}
    for ad ,fn in DONUSUM .items ():
        if fn is None :
            XTR [ad ]=Xtr_ham 
            continue 
        M =np .zeros ((len (Xtr_ham ),Xtr_ham .shape [1 ]*2 ))
        for u in np .unique (tr_pid ):
            i =np .where (tr_pid ==u )[0 ]
            M [i ]=uygula (Xtr_ham [i ],fn )
        XTR [ad ]=M 

    print (f"\n{'split':<24}{'arm':>14}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}")
    R ={}
    for ad ,fn in DONUSUM .items ():
        M =XTR [ad ]
        kur =lambda keep :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,
        n_jobs =NJOB ,random_state =0 ).fit (
        M [keep ],ytr [keep ])

        def puanla (clf ,alt ):
            det ,rob =[],[]
            for r in alt :
                P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
                if r ["X"]is not None :
                    s =clf .predict_proba (uygula (r ["X"],fn ))[:,1 ]
                    m =(s >=ORAN *max (float (s .max ()),1e-9 ))&(s >=TABAN )
                    if m .any ():
                        P =r ["P"][m ];Pd =r ["Pd"][m ]
                k ="cok"if r ["n"]>=8 else "dusuk"
                det .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
                rob .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
            return det ,rob 

        det ,rob =puanla (kur (~np .isin (tr_grp ,list (tg ))),DER )
        p_ ,r_ =pr (det )
        print (f"{'TANIDIK (DEV+VAL)':<24}{ad :>14}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}"
        f"{p_ :>9.3f}{r_ :>9.3f}",flush =True )
        R [(ad ,"tanidik")]=(det ,rob )
        for k ,mad in kod .items ():
            alt =[x for x in DER if x ["mfg"]==mad ]
            if len (alt )<10 :
                continue 
            det2 ,rob2 =puanla (kur ((tr_mfg !=k )&~np .isin (tr_grp ,list (tg ))),alt )
            p2 ,r2 =pr (det2 )
            print (f"{'  '+mad +' disarida ('+str (len (alt ))+')':<24}{ad :>14}"
            f"{f1w (det2 ):>9.4f}{f1w (rob2 ):>9.4f}{p2 :>9.3f}{r2 :>9.3f}",flush =True )
            R [(ad ,mad )]=(det2 ,rob2 )
        print ()

    anahtar =sorted ({k [1 ]for k in R if k [1 ]!="tanidik"})
    t_tan =f1w (R [("A ham","tanidik")][0 ])
    t_kotu =min (f1w (R [("A ham",a )][0 ])for a in anahtar )
    print (f"KARAR (baseline A ham: tanidik {t_tan :.4f} | gorulmemis en kotu {t_kotu :.4f})")
    kazanan =None 
    for ad in DONUSUM :
        if ad =="A ham":
            continue 
        dt =f1w (R [(ad ,"tanidik")][0 ])-t_tan 
        ek =min (f1w (R [(ad ,a )][0 ])for a in anahtar )
        gecti =(dt >=-0.01 )and (ek -t_kotu >=0.01 )
        print (f"  {ad :<14} tanidik {dt :+.4f} | en kotu {t_kotu :.4f} -> {ek :.4f} "
        f"({ek -t_kotu :+.4f}) -> {'GECTI'if gecti else 'GECMEDI'}")
        if gecti and (kazanan is None or ek >min (f1w (R [(kazanan ,a )][0 ])for a in anahtar )):
            kazanan =ad 
    print (f"\nSONUC: {(kazanan +' DAGITILABILIR')if kazanan else 'HICBIRI GECMEDI'}")
    json .dump ({f"{k [0 ]}|{k [1 ]}":{"tespit":float (f1w (v [0 ])),"robot":float (f1w (v [1 ]))}
    for k ,v in R .items ()}|{"kazanan":kazanan },
    open ("results/u4_parca_ici_uctan_uca.json","w"),indent =1 )
    pickle .dump (R ,open ("results/u4_parca.pkl","wb"))
    print ("receipt -> results/u4_parca_ici_uctan_uca.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
