# -*- coding: utf-8 -*-
"""P: robot-hazir WHY DEV'de 0.4974, VAL'de 0.3470?

FINDING (measured): two kumenin TESPIT F1'i cakisiyor (0.7073 [0.6523,0.7603] vs 0.6529
[0.5791,0.7222]) but ROBOT-HAZIR araliklari neredeyse AYRIK (DEV lower boundary 0.4259 > VAL upper
boundary 0.4225). Yani "bulma" two kumede same, "robota verilebilir kalitede bulma" not.

HIPOTEZ: konum/axis kalitesi TESPITTEN DAHA COK geometriye bagli. DEV old kumeydi and
segmentasyon training geometrilerine more yakin; VAL that gruplarin TAMAMEN disindan secildi.

BU BETIK: each two kumede, ESLESMIS (i.e. tespit edilmis) candidates for ham error dagilimini
karsilastirir -- lateral mm and axis derece. Boylece kaybin
   (a) lateral hatadan mi, (b) axis hatasindan mi, (c) ikisinden mi
geldigi ayrisir. Kaldirac hangisinde oldugunu bilmeden konum isine girilmez (M dersi).

Yeni inference YOK: exam kosularinin turetmelerini yeniden kullanmak instead of onbellekten
hizlica yeniden turetir (gate uygulanmis hali not, HAM candidate -> esleme -> error).
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def hatalar (cluster ):
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from j_position_mean import vote_avg 

    cf =f"results/_probs_{cluster }.pkl"
    if not os .path .exists (cf )and cluster =="dev":
        cf ="results/_h_probs.pkl"
    cache =pickle .load (open (cf ,"rb"))
    for r in cache :
        r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"low":float (cfg ["robot_wire_gate_threshold"]),
    "very":float (cfg ["robot_wire_gate_threshold_highcp"])}
    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (r ["pid"],"absent:"+r ["pid"])for r in cache }
    pids =np .array ([str (x )for x in d ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"absent:"+p )for p in pids ]),list (tg ))
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (d ["X"][keep ],d ["y"][keep ])

    lat ,ang ,rej =[],[],[]
    for r in cache :
        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
        mk =lambda pr_ ,**kw :cp_openings .connection_points (
        V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r ["stp"],**kw )
        merge =lambda L :vote_avg (L ,min_votes =1 ,mode ="wmean")
        base =merge ([mk (pb )for pb in plist ])
        is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
        cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
        if not cps :
            continue 
        Xc =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT )
        m =clf .predict_proba (Xc )[:,1 ]>=(THR ["very"]if is_hi else THR ["low"])
        if not m .any ():
            continue 
        P =np .array ([c ["point"]for c ,k in zip (cps ,m )if k ],float )
        Pd =np .array ([c ["direction"]for c ,k in zip (cps ,m )if k ],float )
        G ,Gd =r ["G"],r ["Gd"]
        if not len (G ):
            continue 
            # GEVSEK (tespit) olcutle esle -> only BULUNMUS olanlarin kalitesine bak
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        an =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))
        t0 =max (3.0 ,0.06 *r ["diag"])
        pem =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
        us ,ug =set (),set ()
        for d_ ,a_ ,b_ in sorted ((pem [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if d_ >t0 or a_ in us or b_ in ug :
                continue 
            us .add (a_ );ug .add (b_ )
            lat .append (float (pe [a_ ,b_ ]));ang .append (float (an [a_ ,b_ ]))
            rej .append ("very"if r ["n"]>=8 else "low")
    return np .array (lat ),np .array (ang ),np .array (rej )


def main ():
    out ={}
    print (f"{'cluster':<6}{'n':>6}{'lateral med':>11}{'lateral p90':>11}{'<=2mm':>8}"
    f"{'angle med':>10}{'angle p90':>10}{'<=10d':>8}{'IKISI':>8}")
    D ={}
    for k in ("dev","val"):
        lat ,ang ,rej =hatalar (k )
        D [k ]=(lat ,ang ,rej )
        y2 =float ((lat <=2.0 ).mean ());a10 =float ((ang <=10.0 ).mean ())
        ik =float (((lat <=2.0 )&(ang <=10.0 )).mean ())
        print (f"{k :<6}{len (lat ):>6}{np .median (lat ):>11.2f}{np .percentile (lat ,90 ):>11.2f}"
        f"{y2 :>8.3f}{np .median (ang ):>10.2f}{np .percentile (ang ,90 ):>10.2f}{a10 :>8.3f}{ik :>8.3f}")
        out [k ]={"n":len (lat ),"yanal_med":float (np .median (lat )),
        "yanal_p90":float (np .percentile (lat ,90 )),"yanal_2mm_orani":y2 ,
        "aci_med":float (np .median (ang )),"aci_p90":float (np .percentile (ang ,90 )),
        "aci_10d_orani":a10 ,"ikisi_orani":ik }

    print ("\nKAYBIN AYRISTIRILMASI (DEV -> VAL, esles"
    "mis candidates on):")
    dv ,vv =out ["dev"],out ["val"]
    print (f"  yalniz lateral esigi (<=2mm) : {dv ['yanal_2mm_orani']:.3f} -> {vv ['yanal_2mm_orani']:.3f}"
    f"  ({vv ['yanal_2mm_orani']-dv ['yanal_2mm_orani']:+.3f})")
    print (f"  yalniz aci esigi (<=10 der): {dv ['aci_10d_orani']:.3f} -> {vv ['aci_10d_orani']:.3f}"
    f"  ({vv ['aci_10d_orani']-dv ['aci_10d_orani']:+.3f})")
    print (f"  ikisi birden              : {dv ['ikisi_orani']:.3f} -> {vv ['ikisi_orani']:.3f}"
    f"  ({vv ['ikisi_orani']-dv ['ikisi_orani']:+.3f})")
    dom =("YANAL"if (dv ['yanal_2mm_orani']-vv ['yanal_2mm_orani'])
    >(dv ['aci_10d_orani']-vv ['aci_10d_orani'])else "ACI")
    print (f"\n  -> kaybin BUYUK kismi {dom } tarafinda. Kaldirac orada aranir.")
    out ["baskin"]=dom 

    # regime ayrimi: low-CP mi very-CP mi bozuluyor?
    print (f"\n{'regime':<8}{'DEV ikisi':>11}{'VAL ikisi':>11}{'difference':>9}")
    for rk in ("low","very"):
        v =[]
        for k in ("dev","val"):
            lat ,ang ,rej =D [k ]
            m =rej ==rk 
            v .append (float (((lat [m ]<=2.0 )&(ang [m ]<=10.0 )).mean ())if m .any ()else float ("nan"))
        print (f"{rk :<8}{v [0 ]:>11.3f}{v [1 ]:>11.3f}{v [1 ]-v [0 ]:>+9.3f}")
        out .setdefault ("regime",{})[rk ]=v 

    json .dump (out ,open ("results/p_konum_geometri.json","w"),indent =1 )
    print ("\nmakbuz -> results/p_konum_geometri.json")


if __name__ =="__main__":
    main ()
