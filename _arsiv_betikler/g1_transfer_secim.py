# -*- coding: utf-8 -*-
"""G1: TRANSFER-KARARLI OZNITELIK SECIMI (+ G0 baseline).

DIAGNOSIS (x1 + hafiza): gate EZBERLIYOR. Havuzlanmis 0.7584 but gorulmemis ureticide
0.5968/0.6680. 13 baseline ozellikten only `votes` transfer ediyordu (AUC dususu 0.018,
digerleri 0.12-0.18) and most ezberci UCUNU atmak +0.007 vermisti -- SISTEMATIK as never
denenmedi. Su an 58 ham column present.

YONTEM: each ham sutunun TRANSFER KARARLILIGI = |AUC_ic - AUC_disi|
    AUC_ic  : ureticinin KENDI parcalarinda ayirt etme gucu
    AUC_disi: BASKA ureticinin parcalarinda same sutunun gucu
Dususu large which is column, that ureticinin geometrisine OZEL a kisayol ogrenmistir.
Sutunlari kararliliga according to sirala, most kararli k tanesiyle yeniden egit.

KILL (onceden yazildi): manufacturer-disi ORTALAMA +0.01 gelmezse VE havuzlanmis dusmezse
G1 KAPANIR. Kazanan k, havuzlanmisi 0.005'ten extra dusurmemelidir.
"""
import io 
import json 
import sys 

import numpy as np 

import gate_bench as T 


def auc (v ,pos ):
    """Mann-Whitney AUC. Bagli degerleri DOGRU isler (kisa-path surumu sismisti)."""
    v =np .asarray (v ,float )
    if pos .sum ()==0 or (~pos ).sum ()==0 :
        return 0.5 
    from scipy .stats import rankdata 
    r =rankdata (v )
    n1 =pos .sum ();n0 =(~pos ).sum ()
    return float ((r [pos ].sum ()-n1 *(n1 +1 )/2 )/(n1 *n0 ))


def main ():
    from sklearn .ensemble import RandomForestClassifier 

    D =T .yukle ()
    import measure_set 
    measure_set .rapor_bas (D ["rap"])
    X ,y ,pid ,mfg ,keep =D ["X"],D ["y"],D ["pid"],D ["mfg"],D ["keep"]
    AD =D ["ad"]
    print (f"\negitim: {int (keep .sum ())} candidate / {len (np .unique (pid [keep ]))} part | "
    f"{X .shape [1 ]} ham sutun -> {X .shape [1 ]*2 } donusumlu")

    # --- TRANSFER KARARLILIGI: each manufacturer for ic-vs-disi AUC farki
    ureticiler =[k for k in np .unique (mfg )
    if (keep &(mfg ==k )).sum ()>=500 and (keep &(mfg !=k )).sum ()>=500 ]
    print (f"kararlilik manufacturer sayisi: {len (ureticiler )} ({[D ['kod'][k ]for k in ureticiler ]})")
    dus =np .zeros (X .shape [1 ])
    for j in range (X .shape [1 ]):
        f =[]
        for k in ureticiler :
            i_ic =keep &(mfg ==k );i_di =keep &(mfg !=k )
            a_ic =auc (X [i_ic ,j ],y [i_ic ].astype (bool ))
            a_di =auc (X [i_di ,j ],y [i_di ].astype (bool ))
            f .append (abs (abs (a_ic -.5 )-abs (a_di -.5 )))
        dus [j ]=float (np .mean (f ))
    sira =np .argsort (dus )# most KARARLI before
    print (f"\nEN KARARLI 10 : {[AD [i ]for i in sira [:10 ]]}")
    print (f"EN EZBERCI 10 : {[AD [i ]for i in sira [::-1 ][:10 ]]}")

    def yap (sut ):
        idx =np .array (sorted (sut ))

        def arm (X_ ,y_ ,pid_ ,mfg_ ,kp ,th ):
            import wire_gate 
            Xs =X_ [:,idx ]
            Z =np .zeros ((len (Xs ),Xs .shape [1 ]*2 ))
            for u in np .unique (pid_ ):
                i =np .where (pid_ ==u )[0 ]
                Z [i ]=wire_gate .within_part (Xs [i ],D ["donusum"])
            clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =th ).fit (Z [kp ],y_ [kp ])
            return {"clf":clf ,"n_feat":Z .shape [1 ],"donusum":D ["donusum"],
            "_sut":idx }
        return arm 

        # URUNUN karar yolu full column kumesini bekler -> lower cluster for sarmalayici
    import wire_gate 
    _asil_skor =wire_gate .decision_score 

    def skor_altkume (m ,Xh ):
        if isinstance (m ,dict )and "_sut"in m :
            Xh =np .asarray (Xh ,float )[:,m ["_sut"]]
        return _asil_skor (m ,Xh )
    wire_gate .decision_score =skor_altkume 

    T .baslik (D )
    SON ={};PARCA ={}
    for k in (58 ,50 ,45 ,40 ,35 ,30 ,25 ,20 ,15 ,10 ):
        sut =list (sira [:k ])
        ad =f"k={k }"+("  (TABAN)"if k ==58 else "")
        SON [k ],PARCA [k ]=T .calistir (D ,yap (sut ),ad )

    baseline =SON [58 ]
    print (f"\n{'k':<6}{'havuzlanmis':>13}{'URET-ORT':>11}{'d(pool)':>11}{'d(uret)':>10}")
    for k in SON :
        print (f"{k :<6}{SON [k ]['havuzlanmis']['tespit']:>13.4f}"
        f"{SON [k ]['_URETICI_DISI_ORT']:>11.4f}"
        f"{SON [k ]['havuzlanmis']['tespit']-baseline ['havuzlanmis']['tespit']:>+11.4f}"
        f"{SON [k ]['_URETICI_DISI_ORT']-baseline ['_URETICI_DISI_ORT']:>+10.4f}")

    candidate =[k for k in SON if k !=58 
    and SON [k ]["_URETICI_DISI_ORT"]-baseline ["_URETICI_DISI_ORT"]>=0.01 
    and SON [k ]["havuzlanmis"]["tespit"]-baseline ["havuzlanmis"]["tespit"]>=-0.005 ]
    if candidate :
        en =max (candidate ,key =lambda k :SON [k ]["_URETICI_DISI_ORT"])
        lo ,hi =T .ga (PARCA [58 ],PARCA [en ],"havuzlanmis")
        print (f"\nKAZANAN k={en } | manufacturer-ort {baseline ['_URETICI_DISI_ORT']:.4f} -> "
        f"{SON [en ]['_URETICI_DISI_ORT']:.4f} | havuzlanmis GA[{lo :+.4f},{hi :+.4f}]")
    else :
        en =None 
        print ("\nKILL: hicbir k manufacturer-disi ORT'de +0.01 vermedi -> G1 KAPANDI")
    with io .open ("results/g1_transfer_secim.json","w",encoding ="utf-8")as f :
        json .dump ({"kararlilik":{AD [i ]:float (dus [i ])for i in sira },
        "sonuc":{str (k ):{"havuzlanmis":SON [k ]["havuzlanmis"]["tespit"],
        "uretici_ort":SON [k ]["_URETICI_DISI_ORT"],
        "robot":SON [k ]["havuzlanmis"]["robot"]}for k in SON },
        "kazanan":en ,
        "secili_sutunlar":[AD [i ]for i in sira [:en ]]if en else None },f ,indent =1 )
    print ("receipt -> results/g1_transfer_secim.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
