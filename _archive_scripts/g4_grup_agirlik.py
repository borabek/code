# -*- coding: utf-8 -*-
"""G4: GRUP-AGIRLIKLI EGITIM -- listenin last gate kalemi.

WHY: corpus GEOMETRI IKIZLERIYLE full (1903 part -> 765 grup). Baskin a aile,
own ornek sayisiyla karar yuzeyini dikte ediyor may be. Her geometri grubunu
ESIT agirlikta saymak this baskiyi kaldirir; new bilgi eklemez, VAR OLANI dengeler.

KILL (onceden): manufacturer-disi ORT +0.01 VE havuzlanmis -0.005'ten iyi.
"""
import io ,json 
import numpy as np 
import gate_bench as T 


def main ():
    import measure_set ,wire_gate 
    from sklearn .ensemble import RandomForestClassifier 
    D =T .yukle ();measure_set .rapor_bas (D ["rap"])
    X ,y ,pid =D ["X"],D ["y"],D ["pid"]
    gk =D ["gk"];grup =np .array ([gk .get (p ,"absent:"+p )for p in pid ])
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],D ["donusum"])

    def yap (mod ):
        def arm (X_ ,y_ ,pid_ ,mfg_ ,kp ,th ):
            w =None 
            if mod =="grup":
                _ ,inv ,cnt =np .unique (grup [kp ],return_inverse =True ,return_counts =True )
                w =1.0 /cnt [inv ]
            elif mod =="part":
                _ ,inv ,cnt =np .unique (pid_ [kp ],return_inverse =True ,return_counts =True )
                w =1.0 /cnt [inv ]
            clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =th ).fit (Z [kp ],y_ [kp ],sample_weight =w )
            return {"clf":clf ,"n_feat":Z .shape [1 ],"donusum":D ["donusum"]}
        return arm 

    T .baslik (D )
    SON ,PARCA ={},{}
    for ad ,mod in (("G0 TABAN","absent"),("G4 grup-agirlikli","grup"),
    ("G4 part-agirlikli","part")):
        SON [ad ],PARCA [ad ]=T .calistir (D ,yap (mod ),ad )
    tb =SON ["G0 TABAN"]
    print (f"\n{'arm':<22}{'d(pool)':>11}{'d(uret)':>10}{'karar':>9}")
    kaz =[]
    for ad in SON :
        dh =SON [ad ]["havuzlanmis"]["detection"]-tb ["havuzlanmis"]["detection"]
        du =SON [ad ]["_URETICI_DISI_ORT"]-tb ["_URETICI_DISI_ORT"]
        ok =du >=0.01 and dh >=-0.005 and ad !="G0 TABAN"
        if ok :kaz .append (ad )
        print (f"{ad :<22}{dh :>+11.4f}{du :>+10.4f}{'GECTI'if ok else '':>9}")
    for ad in kaz :
        lo ,hi =T .ga (PARCA ["G0 TABAN"],PARCA [ad ],"havuzlanmis")
        print (f"  {ad }: havuzlanmis GA[{lo :+.4f},{hi :+.4f}]")
    if not kaz :
        print ("\nKILL: agirliklandirma manufacturer-disi +0.01 vermedi -> G4 CLOSED")
    with io .open ("results/g4_grup_agirlik.json","w",encoding ="utf-8")as f :
        json .dump ({ad :{"havuzlanmis":SON [ad ]["havuzlanmis"]["detection"],
        "uretici_ort":SON [ad ]["_URETICI_DISI_ORT"],
        "robot":SON [ad ]["havuzlanmis"]["robot"]}for ad in SON },f ,indent =1 )
    print ("receipt -> results/g4_grup_agirlik.json")


if __name__ =="__main__":
    import traceback 
    try :main ()
    except BaseException :traceback .print_exc ();raise 
