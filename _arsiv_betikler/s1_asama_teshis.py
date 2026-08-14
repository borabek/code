# -*- coding: utf-8 -*-
"""S1: YONU HANGI ASAMA BOZUYOR? (yarik/push-in sinifina giden path)

R4 showed: 45 derece ustu sapan 107 noktanin 107'si, KESIN analitik B-rep ekseni verilse bile
dik kaliyor. Ama this "no sey yapilamaz" demek DEGIL -- direction a ZINCIRDEN geciyor:

    tez normali (v_o - v_s, Abb.44)  ->  channel_axis  ->  YUVARLAMA  ->  normal-kovaryans  ->  B-rep

Zincirin each halkasi a oncekini EZEBILIYOR. Hangi halkanin GT'ye most yakin oldugunu olcmeden
duzeltmek korlemesine becomes. Burada each CP for tum asamalar kaydedilip GT with karsilastiriliyor.

TEZ NOTU: tez §5.3.6/Abb.44 normali `v_o - v_s` as tanimlar VE "bbox eksenlerine hizali"
der -- i.e. YUVARLAMA tezin own kuralidir. Ama manufacturer GT'sinin %19.1'i 10 dereceden extra
egik. Bu measurement, tezin konvansiyonu with real arasindaki acinin NEREDE olustugunu gosterir.

CIKTI: stage basina "GT'ye 10 derece inside" orani + hangi asamanin ORACLE secimi ne kazandirirdi.
"""
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
ASAMALAR =["tez_normali","channel_axis","yuvarlanmis","normal_kovaryans","brep_eksen","son"]


def main ():
    import cp_openings 
    import robot_cp 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 

    with open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
    CL =float (pp ["cluster_mm"])

    cache =[]
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        with open (cf ,"rb")as f :
            for r in pickle .load (f ):
                r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
                cache .append (r )
    print (f"{len (cache )} part",flush =True )

    KAY =[]
    for i_ ,r in enumerate (cache ,1 ):
        if i_ %50 ==0 :
            print (f"  {i_ }/{len (cache )}",flush =True )
        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
            # TEK model yeter: stage karsilastirmasi for oy birlesimi gerekmez, hatta zararlidir
            # (birlesim yonleri ortalar and stage izini breaks).
        pb =np .asarray (r ["pbs"][0 ],np .float64 )
        cp_openings ._ASAMA_IZ =[]
        cps =cp_openings .connection_points (
        V ,F ,pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,step_path =r ["stp"])
        iz =cp_openings ._ASAMA_IZ 
        cp_openings ._ASAMA_IZ =None 
        if not iz :
            continue 
            # Her IZ kaydini EN YAKIN GT'ye bagla (clustering sonrasi CP listesiyle not, iz noktasiyla)
        for z in iz :
            q =z .get ("nokta")
            if q is None :
                continue 
            dd =np .linalg .norm (G -q ,axis =1 )
            b =int (np .argmin (dd ))
            if dd [b ]>max (3.0 ,0.06 *float (r ["diag"])):
                continue # GT'ye eslesmeyen candidate -- direction dogrulugu tanimsiz
            kay ={"pid":r ["pid"],"mesafe":float (dd [b ])}
            for a in ASAMALAR :
                v =z .get (a )
                kay [a ]=(float (np .degrees (np .arccos (np .clip (
                abs (float (np .asarray (v ,float )@Gd [b ])),0 ,1 ))))if v is not None else None )
            KAY .append (kay )

    print (f"\n{len (KAY )} candidate GT'ye eslendi\n")
    print (f"{'asama':<20}{'var':>7}{'<=10 deg':>10}{'medyan':>9}{'>45 deg':>9}")
    ozet ={}
    for a in ASAMALAR :
        v =np .array ([k [a ]for k in KAY if k [a ]is not None ])
        if not len (v ):
            print (f"{a :<20}{0 :>7}{'-':>10}{'-':>9}{'-':>9}")
            continue 
        ozet [a ]={"n":int (len (v )),"on_alti":float ((v <=10 ).mean ()),
        "medyan":float (np .median (v )),"kirkbes_ustu":float ((v >45 ).mean ())}
        print (f"{a :<20}{len (v ):>7}{(v <=10 ).mean ():>10.1%}{np .median (v ):>9.2f}{(v >45 ).mean ():>9.1%}")

    print ("\nORACLE: her CP icin ASAMALARIN EN IYISI secilseydi")
    en_iyi =[]
    for k in KAY :
        v =[k [a ]for a in ASAMALAR if k [a ]is not None ]
        if v :
            en_iyi .append (min (v ))
    en_iyi =np .array (en_iyi )
    last_ =np .array ([k ["son"]for k in KAY if k ["son"]is not None ])
    print (f"  su anki 'son'      : {(last_ <=10 ).mean ():.1%} <=10 deg")
    print (f"  ORACLE (en iyi)    : {(en_iyi <=10 ).mean ():.1%} <=10 deg  "
    f"(+{((en_iyi <=10 ).mean ()-(last_ <=10 ).mean ())*100 :.1f} puan)")

    print ("\nSON asama KOTUYKEN (>45 deg) hangi asama IYIYDI?")
    kotu =[k for k in KAY if k ["son"]is not None and k ["son"]>45 ]
    print (f"  {len (kotu )} kotu CP")
    for a in ASAMALAR [:-1 ]:
        v =[k [a ]for k in kotu if k [a ]is not None ]
        if v :
            v =np .array (v )
            print (f"    {a :<20} n={len (v ):>4} | <=10 deg {(v <=10 ).mean ():>6.1%} | "
            f"medyan {np .median (v ):>6.2f}")
    print ("  -> a stage here belirgin iyiyse, that asamayi KORUMAK duzeltmedir")

    with open ("results/s1_asama.json","w",encoding ="utf-8")as f :
        json .dump ({"n":len (KAY ),"asamalar":ozet ,
        "oracle_on_alti":float ((en_iyi <=10 ).mean ()),
        "son_on_alti":float ((last_ <=10 ).mean ())},f ,indent =1 )
    with open ("results/s1_kayit.pkl","wb")as f :
        pickle .dump (KAY ,f )
    print ("\nmakbuz -> results/s1_asama.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
