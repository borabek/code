# -*- coding: utf-8 -*-
"""U5: part-ici z-skorun TAKASINI kaldirmayi dene (WEI +0.086 but PXC -0.037).

U4 olctu: D (ham+zskor) gorulmemis ureticide most kotu durumu 0.4832 -> 0.5702 cikariyor
(GA [+0.048,+0.126]) but DIGER manufacturer-disi bolmede -0.0374 kaybettiriyor (GA [-0.065,-0.009]).
Ikisi de GERCEK. Yani this a RISK TAKASI, bedava kazanc not.

Takasi kaldirmayi denemeye value, because bedelin kaynagi belli not. Uc hipotez, three arm:

  E  z-skor YALNIZ ezberleyen sutunlara
     Gerekce: olculmus kayit -- 13 sutundan only `votes` genelleşiyor (manufacturer-disi AUC dususu
     0.018; digerleri 0.12-0.18). Genelleşen sutunu HAM birakip only ezberleyenleri part-ici
     baglamla yumusatmak, bilgi kaybini azaltabilir.
  G  TOPLULUK: ham gate with z-skor gate'in olasiliklarinin ortalamasi
     Gerekce: ensemble usually kazancin cogunu korur, kaybin a kismini geri gives.
  H  z-skor YALNIZ fiziksel/topolojik sutunlara (13..21)
     Gerekce: olasilik istatistikleri (0..12) already parcaya according to olceklidir; kayan sey
     fiziksel dagilimlar may be.

KILL (onceden yazili): a arm D'yi degistirebilmek for (a) most kotu manufacturer-disi durumu D
up to iyi must be (>= D - 0.01) VE (b) DIGER manufacturer-disi bolmedeki kaybi D'den EN AZ 0.015
more few must be. Yoksa D kalir and takas OLDUGU GIBI raporlanir.

Turetme onbellege alinir (results/_u4_der.pkl) -- new kollar dakikalar inside denenir.
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
DER_CACHE ="results/_u4_der.pkl"
NJOB =-1 
GENELLESEN =[11 ]# votes -- single genelleşen column (olculmus)
EZBER =[i for i in range (22 )if i not in GENELLESEN ]
FIZTOP =list (range (13 ,22 ))# B-rep fiziksel + topoloji


def _z (sub ,arm =None ):
    k =list (range (sub .shape [1 ]))if arm is None else list (arm )
    s =sub [:,k ]
    sd =s .std (0 )
    return np .where (sd >1e-12 ,(s -s .mean (0 ))/np .where (sd >1e-12 ,sd ,1.0 ),0.0 )


ARM ={
"A ham":None ,
"D zskor tum":lambda X :np .hstack ([X ,_z (X )]),
"E zskor ezber":lambda X :np .hstack ([X ,_z (X ,EZBER )]),
"H zskor fiztop":lambda X :np .hstack ([X ,_z (X ,FIZTOP )]),
}


def turet ():
    if os .path .exists (DER_CACHE ):
        print (f"turetme onbellekten: {DER_CACHE }",flush =True )
        return pickle .load (open (DER_CACHE ,"rb"))
    import cp_openings ,robot_cp ,wire_gate 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
    CL =float (pp ["cluster_mm"])
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    cache =[]
    for k in ("dev","val"):
        cf =f"results/_probs_{k }.pkl"
        if not os .path .exists (cf )and k =="dev":
            cf ="results/_h_probs.pkl"
        for r in pickle .load (open (cf ,"rb")):
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
            cache .append (r )
    DER =[]
    for i ,r in enumerate (cache ,1 ):
        if i %50 ==0 :
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
    pickle .dump (DER ,open (DER_CACHE ,"wb"))
    print (f"turetme onbellege yazildi -> {DER_CACHE }",flush =True )
    return DER 


def main ():
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    ORAN =float (cfg ["gate_goreli_oran"]);TABAN =float (cfg ["gate_goreli_taban"])
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER =turet ()
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],r .get ("mfg","?"))

    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"absent:"+p )for p in tr_pid ])
    tr_mfg =np .array ([str (x )for x in d ["mfg"]])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in tr_pid [tr_mfg ==k ]).most_common (1 )[0 ][0 ]
    for k in np .unique (tr_mfg )}
    tg ={gk .get (r ["pid"],"absent:"+r ["pid"])for r in DER }

    def egit_veri (fn ):
        if fn is None :
            return Xtr 
        M =None 
        for u in np .unique (tr_pid ):
            i =np .where (tr_pid ==u )[0 ]
            v =fn (Xtr [i ])
            if M is None :
                M =np .zeros ((len (Xtr ),v .shape [1 ]))
            M [i ]=v 
        return M 

    rf =lambda M ,keep :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,
    n_jobs =NJOB ,random_state =0 ).fit (M [keep ],ytr [keep ])

    def skorla (clfs ,fns ,r ):
        """Bir parcanin adaylarini (birden extra arm verilirse ORTALAMA olasilikla) skorla."""
        ss =[]
        for clf ,fn in zip (clfs ,fns ):
            X =r ["X"]if fn is None else fn (r ["X"])
            ss .append (clf .predict_proba (X )[:,1 ])
        return np .mean (ss ,0 )

    def puanla (clfs ,fns ,alt ):
        det =[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None :
                s =skorla (clfs ,fns ,r )
                m =(s >=ORAN *max (float (s .max ()),1e-9 ))&(s >=TABAN )
                if m .any ():
                    P =r ["P"][m ];Pd =r ["Pd"][m ]
            det .append (("very"if r ["n"]>=8 else "low",)
            +esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
        return det 

    VERI ={ad :egit_veri (fn )for ad ,fn in ARM .items ()}
    KOL =dict (ARM )
    KOL ["G ensemble(A+D)"]="TOPLULUK"

    print (f"\n{'arm':<18}{'tanidik':>10}{'PXC-disi':>10}{'WEI-disi':>10}{'ORT':>9}{'EN KOTU':>9}")
    S ={}
    for ad in KOL :
        if ad =="G ensemble(A+D)":
            fns =[None ,ARM ["D zskor tum"]]
            mats =[VERI ["A ham"],VERI ["D zskor tum"]]
        else :
            fns =[ARM [ad ]]
            mats =[VERI [ad ]]
        keep =~np .isin (tr_grp ,list (tg ))
        tan =f1w (puanla ([rf (M ,keep )for M in mats ],fns ,DER ))
        dis ={}
        for k ,mad in kod .items ():
            alt =[x for x in DER if x ["mfg"]==mad ]
            if len (alt )<10 :
                continue 
            k2 =(tr_mfg !=k )&~np .isin (tr_grp ,list (tg ))
            dis [mad ]=f1w (puanla ([rf (M ,k2 )for M in mats ],fns ,alt ))
        v =list (dis .values ())
        print (f"{ad :<18}{tan :>10.4f}{dis .get ('PXC',float ('nan')):>10.4f}"
        f"{dis .get ('WEI',float ('nan')):>10.4f}{np .mean (v ):>9.4f}{min (v ):>9.4f}",flush =True )
        S [ad ]={"tanidik":float (tan ),**{f"{a }_disi":float (b )for a ,b in dis .items ()},
        "ort":float (np .mean (v )),"en_kotu":float (min (v ))}

    A ,D =S ["A ham"],S ["D zskor tum"]
    d_kayip =min (0.0 ,D ["PXC_disi"]-A ["PXC_disi"])
    print (f"\nKARAR (D'nin takasi: en kotu {D ['en_kotu']-A ['en_kotu']:+.4f}, "
    f"diger split {d_kayip :+.4f})")
    kazanan =None 
    for ad ,s in S .items ():
        if ad in ("A ham","D zskor tum"):
            continue 
        loss =min (0.0 ,s ["PXC_disi"]-A ["PXC_disi"])
        gecti =(s ["en_kotu"]>=D ["en_kotu"]-0.01 )and (loss -d_kayip >=0.015 )
        print (f"  {ad :<18} en kotu {s ['en_kotu']:.4f} (D {D ['en_kotu']:.4f}) | "
        f"loss {loss :+.4f} (D {d_kayip :+.4f}) -> {'D YERINE GECER'if gecti else 'GECMEZ'}")
        if gecti and (kazanan is None or s ["en_kotu"]>S [kazanan ]["en_kotu"]):
            kazanan =ad 
    print (f"\nSONUC: {kazanan or 'D KALIR -- takas OLDUGU GIBI raporlanir'}")
    with open ("results/u5_takas.json","w",encoding ="utf-8")as f :# see. [[file-tanitici-sizintisi]]
        json .dump (S |{"kazanan":kazanan },f ,indent =1 )
    print ("receipt -> results/u5_takas.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
