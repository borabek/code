# -*- coding: utf-8 -*-
"""G1b: TRANSFER-KARARLI SECIM -- SIZINTISIZ version.

G1'IN KUSURU (own yakaladigim): kararliligi PXC-vs-WEI karsitligindan hesapliyordum,
after full da PXC-disi / WEI-disi olcuyordum. Yani secim, DISARIDA TUTULACAK ureticinin
etiketlerini goruyordu -> manufacturer-disi kazanci IYIMSER cikar. Klasik secim sizintisi.

G1b DUZELTIR: kararlilik each split for YALNIZ O BOLMENIN EGITIM VERISINDEN is computed.
Karsitlik da manufacturer not, GEOMETRI GRUBU yarilanmasidir (manufacturer disaridayken already
single manufacturer kalabiliyor):

    training parcalarini geometri grubuna according to rastgele two yariya bol
    AUC_A, AUC_B = same sutunun two yaridaki ayirt etme gucu
    kararsizlik = mean | |AUC_A-0.5| - |AUC_B-0.5| |   (birkac split uzerinden)

Boylece secim, degerlendirme kumesini HIC gormez.

KILL (onceden): manufacturer-disi ORT +0.01 VE havuzlanmis -0.005'ten iyi. Yoksa G1 kapanir.
"""
import io 
import json 
import sys 

import numpy as np 

import gate_bench as T 

NBOL =6 # kararlilik for grup-yarilanma count


def auc (v ,pos ):
    from scipy .stats import rankdata 
    v =np .asarray (v ,float )
    if pos .sum ()==0 or (~pos ).sum ()==0 :
        return 0.5 
    r =rankdata (v )
    n1 =int (pos .sum ());n0 =int ((~pos ).sum ())
    return float ((r [pos ].sum ()-n1 *(n1 +1 )/2 )/(n1 *n0 ))


def kararsizlik (X ,y ,grup ,kp ,seed =0 ):
    """Sutun basina TRANSFER KARARSIZLIGI -- only `kp` icindeki veriden."""
    rng =np .random .default_rng (seed )
    g =grup [kp ];Xk =X [kp ];yk =y [kp ].astype (bool )
    ug =np .unique (g )
    S =np .zeros ((NBOL ,X .shape [1 ]))
    for b in range (NBOL ):
        kar =rng .permutation (ug )
        A =np .isin (g ,kar [:len (kar )//2 ])
        for j in range (X .shape [1 ]):
            a =auc (Xk [A ,j ],yk [A ]);c =auc (Xk [~A ,j ],yk [~A ])
            S [b ,j ]=abs (abs (a -.5 )-abs (c -.5 ))
    return S .mean (0 )


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    import measure_set 
    import wire_gate 

    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    X ,y ,pid ,mfg ,keep =D ["X"],D ["y"],D ["pid"],D ["mfg"],D ["keep"]
    AD =D ["ad"];gk =D ["gk"]
    grup =np .array ([gk .get (p ,"yok:"+p )for p in pid ])
    print (f"\negitim: {int (keep .sum ())} candidate / {len (np .unique (pid [keep ]))} part "
    f"/ {len (np .unique (grup [keep ]))} geometri grubu")

    # each split for AYRI ranking (that bolmenin training maskesiyle)
    SIRA ={}
    for b ,mk in [("havuzlanmis",None )]+[(m +"-disi",k )for k ,m in D ["kod"].items ()]:
        kp =keep .copy ()
        if mk is not None :
            kp =kp &(mfg !=mk )
        if kp .sum ()<500 :
            continue 
        s =kararsizlik (X ,y ,grup ,kp )
        SIRA [b ]=np .argsort (s )
        print (f"  {b :<14} en kararli: {[AD [i ]for i in SIRA [b ][:6 ]]}")
        print (f"  {'':<14} en ezberci: {[AD [i ]for i in SIRA [b ][::-1 ][:6 ]]}")

    _asil =wire_gate .decision_score 

    def skor_alt (m ,Xh ):
        if isinstance (m ,dict )and "_sut"in m :
            Xh =np .asarray (Xh ,float )[:,m ["_sut"]]
        return _asil (m ,Xh )
    wire_gate .decision_score =skor_alt 

    def yap (k ):
        def arm (X_ ,y_ ,pid_ ,mfg_ ,kp ,th ):
        # BU BOLMENIN own siralamasi (sizintisiz): kp'den turetilmis olani sec
            key_ =None 
            for b ,mk in [("havuzlanmis",None )]+[(m +"-disi",kk )
            for kk ,m in D ["kod"].items ()]:
                test =keep .copy ()
                if mk is not None :
                    test =test &(mfg !=mk )
                if np .array_equal (test ,kp )and b in SIRA :
                    key_ =b ;break 
            rank_ =SIRA [key_ if key_ else "havuzlanmis"]
            idx =np .array (sorted (rank_ [:k ]))
            Xs =X_ [:,idx ]
            Z =np .zeros ((len (Xs ),Xs .shape [1 ]*2 ))
            for u in np .unique (pid_ ):
                i =np .where (pid_ ==u )[0 ]
                Z [i ]=wire_gate .within_part (Xs [i ],D ["donusum"])
            clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =th ).fit (Z [kp ],y_ [kp ])
            return {"clf":clf ,"n_feat":Z .shape [1 ],"donusum":D ["donusum"],"_sut":idx }
        return arm 

    T .baslik (D )
    SON ,PARCA ={},{}
    for k in (58 ,45 ,35 ,28 ,22 ,16 ,12 ):
        SON [k ],PARCA [k ]=T .calistir (D ,yap (k ),f"k={k }"+("  (TABAN)"if k ==58 else ""))

    tb =SON [58 ]
    print (f"\n{'k':<6}{'havuzlanmis':>13}{'URET-ORT':>11}{'d(pool)':>11}{'d(uret)':>10}")
    for k in SON :
        print (f"{k :<6}{SON [k ]['havuzlanmis']['tespit']:>13.4f}"
        f"{SON [k ]['_URETICI_DISI_ORT']:>11.4f}"
        f"{SON [k ]['havuzlanmis']['tespit']-tb ['havuzlanmis']['tespit']:>+11.4f}"
        f"{SON [k ]['_URETICI_DISI_ORT']-tb ['_URETICI_DISI_ORT']:>+10.4f}")
    candidate =[k for k in SON if k !=58 
    and SON [k ]["_URETICI_DISI_ORT"]-tb ["_URETICI_DISI_ORT"]>=0.01 
    and SON [k ]["havuzlanmis"]["tespit"]-tb ["havuzlanmis"]["tespit"]>=-0.005 ]
    en =max (candidate ,key =lambda k :SON [k ]["_URETICI_DISI_ORT"])if candidate else None 
    if en :
        lo ,hi =T .ga (PARCA [58 ],PARCA [en ],"havuzlanmis")
        l2 ,h2 =T .ga (PARCA [58 ],PARCA [en ],"WEI-disi")
        print (f"\nKAZANAN k={en }: manufacturer-ort {tb ['_URETICI_DISI_ORT']:.4f} -> "
        f"{SON [en ]['_URETICI_DISI_ORT']:.4f}")
        print (f"  havuzlanmis GA[{lo :+.4f},{hi :+.4f}] | WEI-disi GA[{l2 :+.4f},{h2 :+.4f}]")
    else :
        print ("\nKILL: sizintisiz secimde hicbir k +0.01 vermedi -> G1 KAPANDI")
    with io .open ("results/g1b_transfer_durust.json","w",encoding ="utf-8")as f :
        json .dump ({"sonuc":{str (k ):{"havuzlanmis":SON [k ]["havuzlanmis"]["tespit"],
        "uretici_ort":SON [k ]["_URETICI_DISI_ORT"],
        "robot":SON [k ]["havuzlanmis"]["robot"]}for k in SON },
        "kazanan":en ,
        "siralama_havuz":[AD [i ]for i in SIRA ["havuzlanmis"]]},f ,indent =1 )
    print ("receipt -> results/g1b_transfer_durust.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
