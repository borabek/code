# -*- coding: utf-8 -*-
"""T12 (t8 turevi): MODEL SINIFI manufacturer-disi UCTAN UCA difference yaratiyor mu?

Aday duzeyinde (t11, DAGITILAN goreli esikle) measured:
    model              tanidik   PXC-disi  WEI-disi  EN KOTU
    RF leaf3 (mevcut)   0.7381    0.7080    0.4222    0.4222
    ExtraTrees leaf3    0.7287    0.6810    0.4586    0.4586
    RF leaf20           0.7199    0.6923    0.4436    0.4436
ExtraTrees most iyi transferi veriyor VE tanidik bedeli butce inside (-0.0093).

AMA kayitli uyari: "ExtraTrees does NOT transfer, manufacturer-disi -0.061" -- that measurement 13 SUTUN and
SABIT ESIK donemindendi; ikisi de degisti. Uctan uca olcmeden hangi hukmun gecerli oldugu bilinemez.

KILL: ExtraTrees, manufacturer-disi UCTAN UCA tespit F1'de RF'i HER IKI bolmede de gecmezse ALINMAZ.

Buraya up to olculenler:
  T1/T2  manufacturer-disi cokus (candidate duzeyi): 0.7422 -> 0.6399 / 0.2799; ~%42'si kalibrasyon
  T3/T6  goreli threshold + baseline (candidate duzeyi): most kotu 0.2799 -> 0.4402 (baseline .20) / 0.4222 (.25)
  T7     tanidik veride UCTAN UCA bedel: baseline .25 -> DEV -0.0140, VAL -0.0022 (ort -0.0081)

EKSIK OLAN: manufacturer-disi kazanc UCTAN UCA dogrulanmadi. Bugun candidate duzeyi two times yanildi
(EK blogu, goreli esigin bedeli), that yuzden karar this betikle veriliyor.

TASARIM: DEV+VAL parcalari URETICIYE according to ayrilir. Her manufacturer for gate, O URETICIYI HIC
GORMEDEN egitilir (also test geometri gruplari da dislanir). Turetme a times; two karar
kurali same skorlar on.

KILL (onceden yazili): goreli threshold, manufacturer-disi UCTAN UCA tespit F1'de sabiti HER IKI
manufacturer de gecmezse DAGITILMAZ.
"""
import os ,sys ,json ,pickle ,collections 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier ,ExtraTreesClassifier 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w ,pr 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    cache =[]
    for k in ("dev","val"):
        cf =f"results/_probs_{k }.pkl"
        if not os .path .exists (cf )and k =="dev":
            cf ="results/_h_probs.pkl"
        for r in pickle .load (open (cf ,"rb")):
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
            cache .append (r )
    say =collections .Counter (mfg_of .get (r ["pid"],"?")for r in cache )
    print (f"{len (cache )} part | manufacturer dagilimi {dict (say )}",flush =True )

    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"dusuk":float (cfg ["robot_wire_gate_threshold"]),
    "cok":float (cfg ["robot_wire_gate_threshold_highcp"])}

    # --- TURETME a times (ozellikler saklanir) ---
    DER =[]
    for _i ,r in enumerate (cache ,1 ):
        if _i %25 ==0 :
            print (f"  turetme {_i }/{len (cache )}",flush =True )
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
        step_path =r ["stp"])[:,:18 ]if cps else None )
        DER .append (dict (pid =r ["pid"],mfg =mfg_of .get (r ["pid"],"?"),X =Xc ,is_hi =is_hi ,
        P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))
    print (f"turetme bitti ({sum (len (x ['P'])for x in DER )} candidate)",flush =True )

    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (x ["pid"],"yok:"+x ["pid"])for x in DER }
    tr_mfg =np .array ([str (x )for x in d ["mfg"]])
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    # training korpusundaki manufacturer kodunu ADA cevir (0/1 -> WEI/PXC)
    kod ={}
    for k in np .unique (tr_mfg ):
        adlar =collections .Counter (mfg_of .get (p ,"?")for p in tr_pid [tr_mfg ==k ])
        kod [k ]=adlar .most_common (1 )[0 ][0 ]
    print (f"training korpusu manufacturer kodlari: {kod }",flush =True )

    print (f"\n{'manufacturer (test)':<18}{'kural':<26}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}")
    out ={}
    for k ,ad in kod .items ():
        icinde =[x for x in DER if x ["mfg"]==ad ]
        if len (icinde )<10 :
            print (f"{ad }: {len (icinde )} part, atlandi");continue 
        keep =(tr_mfg !=k )&~np .isin (tr_grp ,list (tg ))
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (d ["X"][keep ][:,:18 ],d ["y"][keep ])
        ORAN =float (cfg .get ("gate_goreli_oran",0.5 ));TABAN =float (cfg .get ("gate_goreli_taban",0.25 ))
        MOD ={"RF leaf3 (mevcut)":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,
        n_jobs =-1 ,random_state =0 ),
        "ExtraTrees leaf3":ExtraTreesClassifier (n_estimators =400 ,min_samples_leaf =3 ,
        n_jobs =-1 ,random_state =0 ),
        "RF leaf20":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =20 ,
        n_jobs =-1 ,random_state =0 )}
        for mad ,mdl in MOD .items ():
            clf2 =mdl .fit (d ["X"][keep ][:,:18 ],d ["y"][keep ])
            det ,rob =[],[]
            for r in icinde :
                P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
                if r ["X"]is not None :
                    s =clf2 .predict_proba (r ["X"])[:,1 ]
                    m =(s >=ORAN *max (float (s .max ()),1e-9 ))&(s >=TABAN )
                    if m .any ():
                        P =r ["P"][m ];Pd =r ["Pd"][m ]
                kk ="cok"if r ["n"]>=8 else "dusuk"
                det .append ((kk ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
                rob .append ((kk ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
            p_ ,r_ =pr (det )
            nm =mad 
            print (f"{ad +' ('+str (len (icinde ))+')':<18}{nm :<26}{f1w (det ):>9.4f}"
            f"{f1w (rob ):>9.4f}{p_ :>9.3f}{r_ :>9.3f}",flush =True )
            out .setdefault (ad ,{})[nm ]={"tespit":float (f1w (det )),"robot":float (f1w (rob ))}
        print ()

    print ("DECISION:")
    ok =True 
    for ad ,v in out .items ():
        sb =v ["RF leaf3 (mevcut)"]["tespit"]
        for nm ,vv in v .items ():
            if nm =="RF leaf3 (mevcut)":
                continue 
            print (f"  {ad }: {nm } {vv ['tespit']-sb :+.4f}")
        ok =ok and (v ["ExtraTrees leaf3"]["tespit"]>sb )
    print (f"\nKILL: goreli threshold HER IKI manufacturer-disi bolmede sabiti gecmezse DAGITILMAZ -> "
    f"{'GECTI'if ok else 'GECMEDI'}")
    json .dump (out ,open ("results/t12_model_uretici_disi.json","w"),indent =1 )
    print ("receipt -> results/t12_model_uretici_disi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
    # SESSIZ OLUM YASAK: t8'in first kosusu no iz birakmadan became (brep_axes file
    # tanitici sizintisi). Artik each cikis nedeni loga yazilir.
        traceback .print_exc ()
        raise 
