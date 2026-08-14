# -*- coding: utf-8 -*-
"""P5-a: PARCA-DUZEYI YON SECICISI -- single karar, tum adaylara uygulanir.

OLCULMUS DAYANAK (2026-08-06, D6):
  * a parcadaki GT yonleri each other **~%100 PARALEL** (medyan %100, two rejimde de)
    -> part basina TEK correct direction present
  * {+-mesh eksenleri, +-candidate bulutu PCA eksenleri} = 12 option uzerinden KAHIN robot
    0.2060 -> **0.3454** (+0.1394)
  * very-CP'de GT yonu B-rep'te YOK (%57'si tum eksenlere >45 derece) -> B-rep tabanli
    each dal orada tikaniyor; this arm B-rep'e BAGLI DEGIL

WHY ONCEKI UZLASI KOLU DUSTU: that, yonu B-rep eksenlerini SAYARAK ariyordu. Dogru direction
B-rep'te olmadigi for most kalabalik cluster vida deligi/ray yuvasiydi. Iki measurement tutarli.

DECISION PARCA DUZEYINDE: candidate basina not. Boylece single a kotu candidate tum parcayi
bozamaz and ogrenilecek sey very more few parametreli becomes.

OZNITELIKLER: dunya XYZ'sine BAGLI DEGIL -- all of them ya direction adayinin part icindeki
GORELI durusu ya da candidate bulutu istatistigi. Uretici kimligi YOK.

TEZ DEGISMEZ: `v_o` konumu OYNAMAZ, 5 sinif and remesh aynen kalir. Bu a YON
secimidir and tezin ham `v_o - v_s`'si adaylardan biridir.
"""
import argparse 
import collections 
import glob 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import d6_record 

MODEL ="results/p5_part_direction.pkl"
MAKBUZ ="results/p5_part_direction.json"
DEV_MFG ={"SUPU","NIT","S+S","SE"}
OB_D6 ="results/_p1_olasilik"
ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 
OZ_AD =["mevcut_uyum","mevcut_uyum_med","brep_uyum","tur_pca","yayilim_boyu",
"yayilim_dik","yayilim_orani","n_aday","kosegen","eksen_sirasi",
"aday_duzlemsel","uyum_std"]


def yon_adaylari (V ,P ):
    """12 direction adayi: +-mesh PCA eksenleri, +-candidate bulutu PCA eksenleri.

    Dunya eksenleri instead of PCA is used: CAD parcalari most zaman axis-hizali
    modellenir but buna GUVENMEK cerceve bagimliligi yaratir. PCA hem hizali hem
    hizasiz parcada same sonucu gives.
    """
    def pca (Q ):
        if len (Q )<3 :
            return []
        Q =Q -Q .mean (0 )
        try :
            _u ,_s ,vt =np .linalg .svd (Q ,full_matrices =False )
        except np .linalg .LinAlgError :
            return []
        return [vt [i ]for i in range (3 )]
    ax =[(a ,0.0 ,i )for i ,a in enumerate (pca (V ))]# mesh ekseni
    ax +=[(a ,1.0 ,i )for i ,a in enumerate (pca (P ))]# candidate bulutu ekseni
    out =[]
    for a ,tur ,rank_ in ax :
        a =np .asarray (a ,float );n =np .linalg .norm (a )
        if n <1e-9 :
            continue 
        a =a /n 
        out .append ((a ,tur ,rank_ ));out .append ((-a ,tur ,rank_ ))
    return out 


def oznitelik (a ,tur ,rank_ ,P ,D ,V ,brep_ax ,diag ):
    uy =np .abs (D @a )if len (D )else np .array ([0.0 ])
    if len (P )>=2 :
        Q =P -P .mean (0 )
        boy =float (np .abs (Q @a ).max ())
        dik =float (np .linalg .norm (Q -np .outer (Q @a ,a ),axis =1 ).max ())
    else :
        boy =dik =0.0 
    this =0.0 
    if brep_ax is not None and len (brep_ax ):
        this =float (np .mean (np .abs (brep_ax @a )>=np .cos (np .radians (10 ))))
    duz =0.0 
    if len (P )>=4 :
        Q =P -P .mean (0 )
        try :
            _u ,s ,_vt =np .linalg .svd (Q ,full_matrices =False )
            duz =float (s [2 ]/(s [0 ]+1e-9 ))
        except np .linalg .LinAlgError :
            pass 
    return [float (uy .mean ()),float (np .median (uy )),this ,float (tur ),
    boy /max (diag ,1e-6 ),dik /max (diag ,1e-6 ),
    boy /max (dik ,1e-6 ),float (len (P )),diag ,float (rank_ ),duz ,
    float (uy .std ())]


def parca_verisi (rec_ ,gate ,cyl ,ob_kok ,match_hungarian ,sadece =None ):
    """Her part for (feature, label) -- label: this direction EN COK robot-hazir veren mi."""
    import brep_snap 
    import p3c_axis_selector as P3C 
    X ,y ,grp ,ek =[],[],[],[]
    for pid ,r in rec_ .items ():
        if sadece is not None and r ["mfg"]not in sadece :
            continue 
        pak =P3C .parca_adaylari (r ,gate ,cyl )
        if pak is None :
            continue 
        P ,D ,_S ,_k =pak 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G )or not len (P ):
            continue 
        f =os .path .join (ob_kok ,pid +".npz")
        V =np .asarray (np .load (f )["V"],float )if os .path .exists (f )else P 
        BA =np .asarray ([c ["axis"]for c in (cyl .get (pid )or [])
        if brep_snap .R_MIN <=c ["radius"]<=brep_snap .R_MAX ],float )
        ad =yon_adaylari (V ,P )
        if not ad :
            continue 
        score =[]
        for (a ,tur ,rank_ )in ad :
            Dk =np .tile (a ,(len (P ),1 ))
            tp =match_hungarian (P ,Dk ,G ,Gd ,r ["diag"],ROBOT_YANAL ,ROBOT_ACI ,
            False ,signed =True )[0 ]
            score .append (tp )
        en =max (score )
        for (a ,tur ,rank_ ),s in zip (ad ,score ):
            X .append (oznitelik (a ,tur ,rank_ ,P ,D ,V ,BA ,r ["diag"]))
            y .append (int (s ==en and en >0 ))
            grp .append (pid );ek .append ((pid ,a ))
    return np .asarray (X ,float ),np .asarray (y ,int ),np .asarray (grp ,str ),ek 


def uygula (rec_ ,gate ,cyl ,ob_kok ,sec ,threshold ,match_hungarian ,f1w ,mfgler =None ):
    import brep_snap 
    import p3c_axis_selector as P3C 
    T ,R =[],[]
    for pid ,r in rec_ .items ():
        if mfgler is not None and r ["mfg"]not in mfgler :
            continue 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="very"if r ["n"]>=8 else "low"
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        pak =P3C .parca_adaylari (r ,gate ,cyl )
        if pak is not None :
            P ,D ,_S ,_k =pak 
            if sec is not None and len (P ):
                f =os .path .join (ob_kok ,pid +".npz")
                V =np .asarray (np .load (f )["V"],float )if os .path .exists (f )else P 
                BA =np .asarray ([c ["axis"]for c in (cyl .get (pid )or [])
                if brep_snap .R_MIN <=c ["radius"]<=brep_snap .R_MAX ],
                float )
                ad =yon_adaylari (V ,P )
                if ad :
                    Z =np .asarray ([oznitelik (a ,t ,s ,P ,D ,V ,BA ,r ["diag"])
                    for (a ,t ,s )in ad ],float )
                    sk =sec .predict_proba (Z )[:,1 ]
                    j =int (np .argmax (sk ))
                    if sk [j ]>=threshold :
                        D =np .tile (ad [j ][0 ],(len (P ),1 ))
        T .append ((rj ,)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
        R .append ((rj ,)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,ROBOT_ACI ,
        False ,signed =True )[:3 ])
    return f1w (T ),f1w (R )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--gate",default ="results/wire_gate_v5.pkl")
    a =ap .parse_args ()
    import protocol 
    protocol .tez_dogrula ()
    from sklearn .ensemble import RandomForestClassifier 
    from sina_cluster import match_hungarian ,f1w 

    with open (a .gate ,"rb")as f :
        gate =pickle .load (f )
    cyl_egt ={}
    if os .path .exists ("results/_p3c_silindir_egitim.pkl"):
        with open ("results/_p3c_silindir_egitim.pkl","rb")as f :
            cyl_egt =pickle .load (f )
    with open ("results/_d6_silindirler.pkl","rb")as f :
        cyl_d6 =pickle .load (f )

        # EGITIM: training korpusu ureticileri (D6'nin 8'i BURADA YOK)
    d =np .load ("results/zengin_parite_v3.npz",allow_pickle =True )
    ek =d6_record .yukle (set (map (str ,d ["pids"])))
    print (f"EGITIM havuzu: {len (ek )} part")
    X ,y ,grp ,_e =parca_verisi (ek ,gate ,cyl_egt ,"results/_yok_",match_hungarian )
    print (f"  training cifti: {len (y )} | pozitif %{100 *y .mean ():.1f} | part {len (set (grp ))}")
    if len (y )<100 :
        print ("  YETERSIZ VERI -- arm acilmadi");return 
    sec =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ,class_weight ="balanced").fit (X ,y )
    print ("  onem: "+", ".join (f"{n }={v :.3f}"for n ,v in 
    sorted (zip (OZ_AD ,sec .feature_importances_ ),
    key =lambda t :-t [1 ])[:5 ]))
    with open (MODEL ,"wb")as f :
        pickle .dump ({"clf":sec ,"feat":OZ_AD },f )

    sv =d6_record .exam ()
    rec_ =d6_record .yukle (set (sv ["pidler"]))
    dev ={p :r for p ,r in rec_ .items ()if r ["mfg"]in DEV_MFG }
    sin ={p :r for p ,r in rec_ .items ()if r ["mfg"]not in DEV_MFG }
    d0t ,d0r =uygula (dev ,gate ,cyl_d6 ,OB_D6 ,None ,0 ,match_hungarian ,f1w )
    print (f"\nDEV secicisiz: detection {d0t :.4f} robot {d0r :.4f}")
    en ,en_r ,izg =None ,d0r ,{}
    for threshold in (0.0 ,0.3 ,0.5 ,0.7 ):
        tf ,rf =uygula (dev ,gate ,cyl_d6 ,OB_D6 ,sec ,threshold ,match_hungarian ,f1w )
        izg [str (threshold )]={"detection":tf ,"robot":rf }
        print (f"  threshold {threshold :.1f}: detection {tf :.4f} robot {rf :.4f}")
        if rf >en_r :
            en_r ,en =rf ,threshold 
    if en is None :
        print ("\nDEV'de no threshold tabani gecmedi -> KOL CLOSED")
        with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
            json .dump ({"karar":"CLOSED","dev_taban":d0r ,"izgara":izg },f ,indent =1 )
        return 
    s0t ,s0r =uygula (sin ,gate ,cyl_d6 ,OB_D6 ,None ,0 ,match_hungarian ,f1w )
    s1t ,s1r =uygula (sin ,gate ,cyl_d6 ,OB_D6 ,sec ,en ,match_hungarian ,f1w )
    print (f"\n--- SINAV YARISI (TEK ATIS) ---")
    print (f"{'setting':<20}{'TESPIT':>9}{'ROBOT':>9}")
    print (f"{'secicisiz':<20}{s0t :>9.4f}{s0r :>9.4f}")
    print (f"{f'part-direction {en :.1f}':<20}{s1t :>9.4f}{s1r :>9.4f}")
    print (f"{'FARK':<20}{s1t -s0t :>+9.4f}{s1r -s0r :>+9.4f}")
    karar ="DAGIT"if s1r >s0r and s1t >=s0t -0.01 else "GERI AL"
    print (f"\nKARAR: {karar }")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"izgara":izg ,"dev_esik":en ,
        "sinav_secicisiz":{"detection":s0t ,"robot":s0r },
        "sinav_secicili":{"detection":s1t ,"robot":s1r },
        "karar":karar },f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
