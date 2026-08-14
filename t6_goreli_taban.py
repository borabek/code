# -*- coding: utf-8 -*-
"""T6: GORELI rule + MUTLAK TABAN (saf goreli kuralin acigini kapatir).

OPEN: saf `v >= 0.5*max(v)` kurali, parcanin most high skoru 0.2 bile olsa 0.1 ustundekileri
kabul eder -- i.e. HER parcadan mutlaka a sey removes. Korpusta sifir-CP parts present; orada
this rule garanti wrong produces. Sabit esikte this sorun absent (all of them elenir) but sabit threshold new
ureticide very high kaliyor (T2: manufacturer 1'de %10.3 atesleme, real %24.1).

COZUM: goreli karar + DUSUK mutlak baseline. Taban, "this parcada never candidate absent" durumunu korur;
goreli kisim distribution kaymasini emer.

KILL (onceden yazili, T3/T4 with same):
  (a) manufacturer-disi HER IKI bolmede sabit esigi gecmeli,
  (b) tanidik veride loss 0.02'yi asmamali.
"""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 


def sk (m ,yy ):
    tp =int ((yy &m ).sum ());fp =int ((~yy &m ).sum ());fn =int ((yy &~m ).sum ())
    pr =tp /max (tp +fp ,1 );rc =tp /max (tp +fn ,1 )
    return 2 *pr *rc /max (pr +rc ,1e-9 ),pr ,rc ,float (m .mean ())


def main ():
    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    X =d ["X"][:,:18 ];y =d ["y"].astype (bool )
    mfg =np .array ([str (x )for x in d ["mfg"]]);pids =np .array ([str (x )for x in d ["pids"]])
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    grp =np .array ([gk .get (p ,"yok:"+p )for p in pids ])
    THR =float (json .load (open ("cp_config.json",encoding ="utf-8"))["robot_wire_gate_threshold"])

    def rule_ (s ,p ,ratio ,baseline ):
        m =np .zeros (len (s ),bool )
        for u in np .unique (p ):
            i =p ==u 
            v =s [i ]
            m [i ]=(v >=ratio *max (v .max (),1e-9 ))&(v >=baseline )
        return m 

    ADAY =[("sabit (mevcut)",None ,None )]+[
    (f"goreli {o } + baseline {t }",o ,t )
    for o in (0.5 ,0.6 )for t in (0.10 ,0.15 ,0.20 ,0.25 )]

    # A) TANIDIK (geometri-disi)
    o_ =np .zeros (len (y ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,grp ):
        o_ [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
    tanidik ={}
    for ad ,orn ,tb in ADAY :
        m =(o_ >=THR )if orn is None else rule_ (o_ ,pids ,orn ,tb )
        tanidik [ad ]=sk (m ,y )

        # B) URETICI-DISI
    mout ={}
    for u in sorted (set (mfg )):
        te =mfg ==u 
        s =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [~te ],y [~te ]).predict_proba (X [te ])[:,1 ]
        for ad ,orn ,tb in ADAY :
            m =(s >=THR )if orn is None else rule_ (s ,pids [te ],orn ,tb )
            mout .setdefault (ad ,{})[u ]=sk (m ,y [te ])

    U =sorted (set (mfg ))
    taban_f =tanidik ["sabit (mevcut)"][0 ]
    print (f"{'kural':<26}{'TANIDIK':>9}{'fark':>8}"+"".join (f"{'uret.'+u :>10}"for u in U )
    +f"{'EN KOTU':>9}{'karar':>9}")
    out ={}
    for ad ,_ ,_ in ADAY :
        t =tanidik [ad ][0 ]
        mv =[mout [ad ][u ][0 ]for u in U ]
        gecti =(ad !="sabit (mevcut)"
        and all (mout [ad ][u ][0 ]>mout ["sabit (mevcut)"][u ][0 ]for u in U )
        and (taban_f -t )<=0.02 )
        print (f"{ad :<26}{t :>9.4f}{t -taban_f :>+8.4f}"+"".join (f"{x :>10.4f}"for x in mv )
        +f"{min (mv ):>9.4f}{('GECTI'if gecti else '-'):>9}")
        out [ad ]={"tanidik":t ,"tanidik_fark":t -taban_f ,
        "manufacturer":{u :mout [ad ][u ][0 ]for u in U },"en_kotu":min (mv ),"gecti":gecti }
    print ("\nKILL: (a) her iki manufacturer-disi bolmede sabiti gecmeli, (b) tanidik loss <= 0.02")
    json .dump (out ,open ("results/t6_goreli_taban.json","w"),indent =1 )
    print ("receipt -> results/t6_goreli_taban.json")


if __name__ =="__main__":
    main ()
