# -*- coding: utf-8 -*-
"""T15: TOPOLOJI blogu UCTAN UCA -- hem TANIDIK hem GORULMEMIS URETICI.

Aday duzeyinde measured:
  t13: kon_cevre AUC 0.709 (TP medyan 1.000 = full kind icbukey halka, FP 0.833), kon_sayi 0.703
  t14: 18 -> 18+4  tanidik +0.0167 | gorulmemis manufacturer EN KOTU +0.0176 (ikisi de POZITIF)

Bu gece two times candidate duzeyi yanildi (EK blogu; goreli esigin bedeli), that yuzden karar here.

TASARIM: candidates BIR KEZ turetilir, 22 column a times is computed; 18-sutunlu arm same matrisin
first 18 sutununu kullanir. Karar kurali each two kolda da DAGITILAN goreli threshold. Tek degisken:
gate'in egitildigi feature kumesi.

KILL (onceden yazili): TOPO, (a) tanidik DEV+VAL tespit F1'i DUSURMEMELI (<=0.01 loss tolere)
VE (b) gorulmemis manufacturer EN KOTU durumunu ARTIRMALI. Ikisi birden saglanmazsa DAGITILMAZ.
"""
import os ,sys ,json ,pickle ,collections 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w ,pr 

    assert len (wire_gate .FEAT_NAMES )==22 ,len (wire_gate .FEAT_NAMES )
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
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
    print (f"{len (cache )} part | {dict (collections .Counter (mfg_of .get (r ['pid'],'?')for r in cache ))}",
    flush =True )

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
        Xc =(wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,
        step_path =r ["stp"])if cps else None )
        DER .append (dict (pid =r ["pid"],cluster =r ["cluster"],mfg =mfg_of .get (r ["pid"],"?"),
        X =Xc ,is_hi =is_hi ,
        P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))
    fb =wire_gate .fallback_ozet ()
    print (f"turetme bitti | notr-donus: {fb if fb else 'YOK'}",flush =True )

    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    tr_mfg =np .array ([str (x )for x in d ["mfg"]])
    kod ={}
    for k in np .unique (tr_mfg ):
        kod [k ]=collections .Counter (mfg_of .get (p ,"?")for p in tr_pid [tr_mfg ==k ]).most_common (1 )[0 ][0 ]

    def kur (keep ,nc ):
        return RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (d ["X"][keep ][:,:nc ],d ["y"][keep ])

    def puanla (clf ,alt ,nc ):
        det ,rob =[],[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None :
                s =clf .predict_proba (r ["X"][:,:nc ])[:,1 ]
                m =(s >=ORAN *max (float (s .max ()),1e-9 ))&(s >=TABAN )
                if m .any ():
                    P =r ["P"][m ];Pd =r ["Pd"][m ]
            k ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    tg ={gk .get (r ["pid"],"yok:"+r ["pid"])for r in DER }
    print (f"\n{'split':<22}{'sutun':>6}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}")
    R ={}
    for nc ,ad in ((18 ,"18 (mevcut)"),(22 ,"22 (+topoloji)")):
        keep =~np .isin (tr_grp ,list (tg ))
        clf =kur (keep ,nc )
        det ,rob =puanla (clf ,DER ,nc )
        p_ ,r_ =pr (det )
        print (f"{'TANIDIK (DEV+VAL)':<22}{nc :>6}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}{p_ :>9.3f}{r_ :>9.3f}",
        flush =True )
        R [(nc ,"tanidik")]=(det ,rob )
        for k ,mad in kod .items ():
            alt =[x for x in DER if x ["mfg"]==mad ]
            if len (alt )<10 :
                continue 
            keep2 =(tr_mfg !=k )&~np .isin (tr_grp ,list (tg ))
            det2 ,rob2 =puanla (kur (keep2 ,nc ),alt ,nc )
            p2 ,r2 =pr (det2 )
            print (f"{'  '+mad +' disarida ('+str (len (alt ))+')':<22}{nc :>6}"
            f"{f1w (det2 ):>9.4f}{f1w (rob2 ):>9.4f}{p2 :>9.3f}{r2 :>9.3f}",flush =True )
            R [(nc ,mad )]=(det2 ,rob2 )
        print ()

    print ("DECISION:")
    key_ =sorted ({k [1 ]for k in R })
    ok_tan =f1w (R [(22 ,"tanidik")][0 ])-f1w (R [(18 ,"tanidik")][0 ])
    mout =[(a ,f1w (R [(22 ,a )][0 ])-f1w (R [(18 ,a )][0 ]))for a in key_ if a !="tanidik"]
    for a ,v in [("tanidik",ok_tan )]+mout :
        print (f"  {a :<16}{v :+.4f}")
    en_kotu18 =min (f1w (R [(18 ,a )][0 ])for a ,_ in mout )
    en_kotu22 =min (f1w (R [(22 ,a )][0 ])for a ,_ in mout )
    gecti =(ok_tan >=-0.01 )and (en_kotu22 >en_kotu18 )
    print (f"\n  gorulmemis manufacturer EN KOTU: {en_kotu18 :.4f} -> {en_kotu22 :.4f} ({en_kotu22 -en_kotu18 :+.4f})")
    print (f"KILL: tanidik >= -0.01 VE en kotu artmali -> {'GECTI'if gecti else 'GECMEDI'}")
    pickle .dump (R ,open ("results/t15_parca.pkl","wb"))
    json .dump ({f"{k [0 ]}|{k [1 ]}":{"tespit":float (f1w (v [0 ])),"robot":float (f1w (v [1 ]))}
    for k ,v in R .items ()}|{"karar":"GECTI"if gecti else "GECMEDI"},
    open ("results/t15_topo_uctan_uca.json","w"),indent =1 )
    print ("receipt -> results/t15_topo_uctan_uca.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ()
        raise 
