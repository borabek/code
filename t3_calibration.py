# -*- coding: utf-8 -*-
"""T3 TEDAVI-1: SABIT threshold instead of DAGILIMA UYARLANAN threshold.

T2 OLCTU: manufacturer-disi cokusun kalibrasyon payi manufacturer 0'da -0.0543, manufacturer 1'de -0.1949.
Kanit: manufacturer 1'de model adaylarin %10.3'une pozitif diyor, real %24.1 -- sabit 0.40 that
dagilimda very high kaliyor (orada most iyi threshold 0.15).

ADAY KURALLAR (none of them new ureticinin istatistigini BILMIYOR):
  sabit        : mevcut urun (threshold cp_config'den)
  ratio         : esigi, EGITIM korpusunun pozitif oranini yakalayacak sekilde test dagiliminda
                 quantile with sec (test etiketlerini KULLANMAZ, only skor dagilimini)
  parca_orani  : each PARCA inside, that parcanin most high skorunun f katindan buyukleri tut
  parca_z      : each PARCA inside skorlari z-skorla, sabit z esigi uygula

KILL (onceden yazili): a rule, HER IKI manufacturer-disi bolmede de sabit esigi gecmezse
ALINMAZ. Tek bolmede kazanip otekinde kaybeden rule kalibrasyon not, sanstir.
"""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from t1_uretici_disi import f1_at 


def main ():
    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    X =d ["X"][:,:18 ];y =d ["y"].astype (bool )
    mfg =np .array ([str (x )for x in d ["mfg"]])
    pids =np .array ([str (x )for x in d ["pids"]])
    THR =float (json .load (open ("cp_config.json",encoding ="utf-8"))["robot_wire_gate_threshold"])

    def parca_kural (s ,p ,fn ):
        """each part inside bagimsiz karar -> maske"""
        m =np .zeros (len (s ),bool )
        for u in np .unique (p ):
            i =p ==u 
            m [i ]=fn (s [i ])
        return m 

    print (f"{'split':<22}{'rule':<16}{'F1':>8}{'kesin':>8}{'recall':>8}{'pozitif%':>10}")
    out ={}
    for u in sorted (set (mfg )):
        te =mfg ==u 
        egit_oran =float (y [~te ].mean ())# EGITIM korpusunun pozitif orani (bilinir)
        s =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [~te ],y [~te ]).predict_proba (X [te ])[:,1 ]
        yy =y [te ];pp =pids [te ]
        kur ={
        "sabit (mevcut)":s >=THR ,
        "ratio-esleme":s >=np .quantile (s ,1.0 -egit_oran ),
        "parca_orani 0.5":parca_kural (s ,pp ,lambda v :v >=0.5 *max (v .max (),1e-9 )),
        "parca_orani 0.6":parca_kural (s ,pp ,lambda v :v >=0.6 *max (v .max (),1e-9 )),
        "parca_z >= 0.0":parca_kural (s ,pp ,lambda v :(v -v .mean ())/(v .std ()+1e-9 )>=0.0 ),
        }
        for ad ,m in kur .items ():
            tp =int ((yy &m ).sum ());fp =int ((~yy &m ).sum ());fn_ =int ((yy &~m ).sum ())
            pr =tp /max (tp +fp ,1 );rc =tp /max (tp +fn_ ,1 )
            f =2 *pr *rc /max (pr +rc ,1e-9 )
            print (f"{'manufacturer '+u +' disarida':<22}{ad :<16}{f :>8.4f}{pr :>8.3f}{rc :>8.3f}"
            f"{float (m .mean ()):>10.3f}")
            out .setdefault (ad ,{})[u ]=f 
        print ()

    U =sorted (set (mfg ))
    baseline =out ["sabit (mevcut)"]
    print (f"{'rule':<18}"+"".join (f"{'manufacturer '+u :>12}"for u in U )+f"{'EN KOTU':>10}{'karar':>10}")
    for ad ,v in out .items ():
        dl =[v [u ]-baseline [u ]for u in U ]
        gecti =all (x >0 for x in dl )
        print (f"{ad :<18}"+"".join (f"{v [u ]:>12.4f}"for u in U )
        +f"{min (v .values ()):>10.4f}"
        +f"{('GECTI'if gecti and ad !='sabit (mevcut)'else '-'):>10}")
    print ("\nKILL: a rule HER IKI bolmede de sabiti gecmezse ALINMAZ.")
    json .dump (out ,open ("results/t3_calibration.json","w"),indent =1 )
    print ("receipt -> results/t3_calibration.json")


if __name__ =="__main__":
    main ()
