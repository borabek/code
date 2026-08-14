# -*- coding: utf-8 -*-
"""G2 + G3: PARCA-ICI DONUSUM and MODEL SINIFI -- ezbere two dogrudan saldiri.

G2 DONUSUM. Su an part-ici Z-SKOR. Z-skor OLCEGE bagimlidir: a manufacturer ailesinin
mutlak buyuklukleri digerinden farkliysa z-skor da kayar. SIRA (rank) donusumu scale-serbesttir
-> manufacturer gecisinde more iyi transfer beklenir. Bu, YENI BILGI eklemez; VAR OLAN bilgiyi
ureticiden bagimsiz kodlar. Tam da ezberin panzehiri.

G3 MODEL SINIFI. ExtraTrees havuzda +0.012 kazanip URETICI-DISINDA -0.061 kaybetmisti
([[extratrees-does-not-transfer]]) -- i.e. model sinifi transferi BELIRLIYOR and this axis
only a times, wrong olcutle was tried. Denenecekler:
    RF depth-sinirli (ezberi dogrudan kisitlar)
    GBM (slow ogrenme + siglik = duzenli)
    L2-lojistik (dogrusal = most high dis-sample kararliligi)
Hepsi AYNI ozniteliklerle; degisen single sey karar yuzeyinin karmasikligi.

YARGI: URETICI-DISI ORTALAMA birincil, havuzlanmis ikincil (ExtraTrees dersi).
KILL: manufacturer-disi +0.01 VE havuzlanmis -0.005'ten iyi degilse arm kapanir.
"""
import io 
import json 

import numpy as np 

import gate_bench as T 


def main ():
    import measure_set 
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier ,HistGradientBoostingClassifier 
    from sklearn .linear_model import LogisticRegression 
    from sklearn .preprocessing import StandardScaler 
    from sklearn .pipeline import make_pipeline 

    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    X ,y ,pid =D ["X"],D ["y"],D ["pid"]
    print (f"\negitim: {int (D ['keep'].sum ())} candidate | donusum={D ['donusum']}")

    def donustur (X_ ,pid_ ,mod ):
        if mod =="ham":
            return X_ .copy ()
        Z =np .zeros ((len (X_ ),X_ .shape [1 ]*2 ))
        for u in np .unique (pid_ ):
            i =np .where (pid_ ==u )[0 ]
            Z [i ]=wire_gate .within_part (X_ [i ],mod )
        return Z 

    ONBELLEK ={}

    def al (mod ):
        if mod not in ONBELLEK :
            ONBELLEK [mod ]=donustur (X ,pid ,mod )
        return ONBELLEK [mod ]

    def yap (mod ,model_ad ):
        def arm (X_ ,y_ ,pid_ ,mfg_ ,kp ,th ):
            Z =al (mod )
            if model_ad =="rf":
                c =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,
                n_jobs =-1 ,random_state =th )
            elif model_ad =="rf_sig":
                c =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,max_depth =8 ,
                max_features =0.3 ,n_jobs =-1 ,random_state =th )
            elif model_ad =="gbm":
                c =HistGradientBoostingClassifier (max_depth =4 ,learning_rate =0.06 ,
                max_iter =300 ,l2_regularization =1.0 ,
                random_state =th )
            elif model_ad =="lojistik":
                c =make_pipeline (StandardScaler (),
                LogisticRegression (C =0.1 ,max_iter =2000 ,random_state =th ))
            c .fit (Z [kp ],y_ [kp ])
            return {"clf":c ,"n_feat":Z .shape [1 ],"donusum":mod if mod !="ham"else None }
        return arm 

    T .baslik (D )
    SON ,PARCA ={},{}
    KOLLAR =[("G0 TABAN  zskor+rf","zskor","rf")]
    for mod in ("sira","ham"):
        KOLLAR .append ((f"G2 {mod }+rf",mod ,"rf"))
    for m in ("rf_sig","gbm","lojistik"):
        KOLLAR .append ((f"G3 zskor+{m }","zskor",m ))
        # most umutlu caprazlama: order donusumu + duzenli model
    KOLLAR +=[("G2xG3 sira+gbm","sira","gbm"),("G2xG3 sira+lojistik","sira","lojistik")]

    for ad ,mod ,m in KOLLAR :
        try :
            SON [ad ],PARCA [ad ]=T .calistir (D ,yap (mod ,m ),ad )
        except Exception as e :
            print (f"{ad :<26} HATA {type (e ).__name__ }: {str (e )[:60 ]}")

    tb =SON ["G0 TABAN  zskor+rf"]
    print (f"\n{'arm':<24}{'pool':>9}{'URET-ORT':>10}{'d(pool)':>10}{'d(uret)':>9}{'karar':>8}")
    kazanan =[]
    for ad in SON :
        dh =SON [ad ]["havuzlanmis"]["tespit"]-tb ["havuzlanmis"]["tespit"]
        du =SON [ad ]["_URETICI_DISI_ORT"]-tb ["_URETICI_DISI_ORT"]
        ok =(du >=0.01 )and (dh >=-0.005 )
        if ok and ad !="G0 TABAN  zskor+rf":
            kazanan .append (ad )
        print (f"{ad :<24}{SON [ad ]['havuzlanmis']['tespit']:>9.4f}"
        f"{SON [ad ]['_URETICI_DISI_ORT']:>10.4f}{dh :>+10.4f}{du :>+9.4f}"
        f"{'GECTI'if ok and ad !='G0 TABAN  zskor+rf'else '':>8}")
    for ad in kazanan :
        lo ,hi =T .ga (PARCA ["G0 TABAN  zskor+rf"],PARCA [ad ],"havuzlanmis")
        l2 ,h2 =T .ga (PARCA ["G0 TABAN  zskor+rf"],PARCA [ad ],"WEI-disi")
        print (f"  {ad }: pool GA[{lo :+.4f},{hi :+.4f}] | WEI-disi GA[{l2 :+.4f},{h2 :+.4f}]")
    if not kazanan :
        print ("\nKILL: hicbir donusum/model kolu manufacturer-disi +0.01 vermedi")
    with io .open ("results/g2g3_donusum_model.json","w",encoding ="utf-8")as f :
        json .dump ({ad :{"havuzlanmis":SON [ad ]["havuzlanmis"]["tespit"],
        "uretici_ort":SON [ad ]["_URETICI_DISI_ORT"],
        "robot":SON [ad ]["havuzlanmis"]["robot"],
        **{b :SON [ad ][b ]["tespit"]for b in SON [ad ]if b .endswith ("-disi")}}
        for ad in SON },f ,indent =1 )
    print ("receipt -> results/g2g3_donusum_model.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
