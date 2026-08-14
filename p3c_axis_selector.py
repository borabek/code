# -*- coding: utf-8 -*-
"""P3-POZ-C: OGRENILMIS EKSEN SECICISI -- kahinin +0.1820'sini almaya calis.

KAHIN OLCUMU (temiz exam): correct (konum, axis) cifti B-rep adaylari arasindan
secilebilseydi robot 0.2060 -> **0.3880**. Yani bilgi VAR, secim YOK.

DUSEN BASIT KURALLAR (ikisi de measured, tekrar acilmayacak):
  * yakinlik-oncelikli oturtma      -> gorulmemiste NET -30 (`r13`)
  * part-ici axis UZLASISI        -> 0.2089 -> 0.1773 (`p3b`)
Ikisi de single a sinyale guveniyordu. Secici BIRDEN COK zayif sinyali merges.

ADAY UZAYI: each candidate for {MEVCUT (oldugu like kal)} + agzi yakin each uygun
silindirin (mouth, +axis) and (mouth, -axis) secenekleri.

OZNITELIKLER URETICI KIMLIGI ICERMEZ -- all of them geometrik/istatistikseldir, otherwise
[[gate-memorizes-not-learns]] tuzagina duseriz.

EGITIM/OLCUM AYRIMI: selector YALNIZ training korpusu ureticilerinde (TOGI/PXC/WEI/SIE/
TE...) egitilir; sinavin 8 ureticisi orada YOKTUR. Esik DEV yarisinda secilir,
SINAV yarisinda TEK ATIS olculur.

TEZ DEGISMEZ: `v_o` turetmesi, 5 sinif and ~6000 remesh aynen kalir. Bu a SON ISLEM
secimidir; ham `v_o` yan yana raporlanabilir.
"""
import argparse 
import collections 
import glob 
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import d6_record 

SILINDIR_EGT ="results/_p3c_silindir_egitim.pkl"
SILINDIR_SIN ="results/_d6_silindirler.pkl"
MODEL ="results/p3c_axis_selector.pkl"
MAKBUZ ="results/p3c_axis_selector.json"
DEV_MFG ={"SUPU","NIT","S+S","SE"}
ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 
MM_MAX =8.0 # candidate uzayina alinacak most large mouth mesafesi
OZ_AD =["aci_mevcut","mesafe","mesafe_norm","yaricap","uzunluk","mevcut_mu",
"paralel_sayi","paralel_pay","eksen_hiza","komsu_uyum","gate_skoru",
"n_aday","n_silindir",
]

# --- K1.7: YEREL goreli radius (OPT-IN, VARSAYILAN KAPALI) ---------------
# Fikir: ham "radius" ureticiler arasi kiyaslanamaz (tel girisleri 0.1-3.8mm
# arasi each places), ayirt edici which is adayin komsulugundaki DIGER silindirlere
# according to buyuklugu. Sonda destekliyordu: correct silindir komsulugun most buyugu
# %54.2, first ikisinde %87.8 (D6, 640 eslesme).
#
# AMA TEMIZ A/B NULL CIKTI (same 2584 part, same measurement, single degisken feature):
#     13 feature -> DEV 0.3035 / SINAV 0.2676
#     17 feature -> DEV 0.3122 / SINAV 0.2621
# Ayar kumesinde +0.0087, AYRIK kumede -0.0055 = klasik fake kazanc deseni.
# NOT DEPLOYED. Kod duruyor because LOMO secimi (SIRA-7) ya da different a
# birlestirmeyle yeniden denenebilir; but urun yolunun bedava hesap yapmamasi
# for varsayilan KAPALI.
P3C_YEREL_YARICAP =os .environ .get ("P3C_YEREL_YARICAP","0")=="1"
if P3C_YEREL_YARICAP :
    OZ_AD =OZ_AD +["yaricap_orani","yaricap_sira","yerel_en_buyuk","n_yerel"]


def _uyumlu (sec ,X ):
    """Ozellik genisligini modelin bekledigi genislige DARALT.

    K1.7 with OZ_AD 13 -> 17'ye output. Eski ckpt'ler (p3c_axis_selector.pkl) 13
    bekliyor. Genislik sessizce uyusmazsa sklearn ya patlar ya da -- more kotusu --
    baska a places sessizce wrong skor produces. Yeni ozellikler LISTENIN SONUNA
    eklendigi for first n column old sirayla same; bastan kesmek GUVENLI.
    Genisletme YAPILMAZ: model more genis bekliyorsa this a hatadir, patlasin.
    """
    n =int (getattr (sec ,"n_features_in_",X .shape [1 ]))
    if X .shape [1 ]==n :
        return X 
    if X .shape [1 ]<n :
        raise ValueError (
        f"p3c ozellik genisligi {X .shape [1 ]} < modelin bekledigi {n }: "
        "model with kod uyumsuz, refit gerekli"
        )
    return X [:,:n ]


def _yerel_yaricaplar (uy ,p ):
    """Adayin MM_MAX komsulugunda mouth veren silindirlerin yaricaplari."""
    out =[]
    for c in uy :
        for m in (c ["mouth_a"],c ["mouth_b"]):
            if float (np .linalg .norm (np .asarray (m ,float )-p ))<=MM_MAX :
                out .append (float (c ["radius"]))
                break 
    return np .asarray (out ,float )


def secenekler (cyls ,p ,d ,diag ,gate_s ,komsu ,n_aday ):
    """Bu candidate for (konum, axis, feature) listesi. Ilk oge HER ZAMAN MEVCUT."""
    import brep_snap 
    p =np .asarray (p ,float );d =np .asarray (d ,float )
    uy =[c for c in (cyls or [])
    if brep_snap .R_MIN <=c ["radius"]<=brep_snap .R_MAX ]
    A =np .asarray ([c ["axis"]for c in uy ],float )if uy else np .zeros ((0 ,3 ))

    yerel =_yerel_yaricaplar (uy ,p )if P3C_YEREL_YARICAP else np .zeros (0 )
    r_max =float (yerel .max ())if len (yerel )else 0.0 

    def oz (pp ,aa ,mevcut ,yar ,uzn ,mes ):
        aci =np .degrees (np .arccos (np .clip (abs (float (aa @d )),-1 ,1 )))
        par =int ((np .abs (A @aa )>=np .cos (np .radians (10 ))).sum ())if len (A )else 0 
        if len (yerel )and not mevcut :
            ratio =yar /max (r_max ,1e-6 )
            rank_ =float ((yerel >yar ).sum ())/max (len (yerel ),1 )
            enb =float (yar >=r_max -1e-9 )
        else :
            ratio ,rank_ ,enb =0.0 ,1.0 ,0.0 
        v =[aci ,mes ,mes /max (diag ,1e-6 ),yar ,uzn ,float (mevcut ),
        par ,par /max (len (A ),1 ),float (np .max (np .abs (aa ))),
        abs (float (aa @komsu ))if komsu is not None else 0.0 ,
        gate_s ,n_aday ,len (A )]
        if P3C_YEREL_YARICAP :
            v +=[ratio ,rank_ ,enb ,len (yerel )]
        return v 

    out =[(p ,d ,oz (p ,d ,True ,0.0 ,0.0 ,0.0 ))]
    for c in uy :
        uzn =float (np .linalg .norm (np .asarray (c ["mouth_b"])-np .asarray (c ["mouth_a"])))
        for m in (c ["mouth_a"],c ["mouth_b"]):
            m =np .asarray (m ,float )
            mes =float (np .linalg .norm (m -p ))
            if mes >MM_MAX :
                continue 
            a =np .asarray (c ["axis"],float )
            for s in (1.0 ,-1.0 ):
                out .append ((m ,s *a ,oz (m ,s *a ,False ,c ["radius"],uzn ,mes )))
    return out 


def parca_adaylari (r ,model ,ob ,ratio =0.40 ,baseline =0.30 ):
    """Gate'ten gecen candidates + part duzeyi baglam."""
    import wire_gate 
    from p1c_threshold import maske 
    M =d6_record .x58 (r )
    if M is None or r .get ("P")is None or not len (r ["P"])or M .shape [1 ]*2 !=model ["n_feat"]:
        return None 
    s =np .asarray (wire_gate .decision_score (model ,M ),float )
    k =maske (s ,ratio ,baseline )
    if not k .any ():
        return None 
    P =np .asarray (r ["P"],float )[k ];D =np .asarray (r ["Pd"],float )[k ];S =s [k ]
    komsu =None 
    if len (D )>1 :
        B =D *np .sign (D @D [0 ])[:,None ]
        komsu =B .mean (0 );komsu /=(np .linalg .norm (komsu )+1e-12 )
    return P ,D ,S ,komsu 


def silindir_onbellek (pidler ,yol ):
    import brep_snap 
    from korpus_kimlik import step_kimlik as SK 
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    ob ={}
    if os .path .exists (yol ):
        with open (yol ,"rb")as f :
            ob =pickle .load (f )
    eksik =[p for p in pidler if p not in ob and p in S ]
    if eksik :
        print (f"silindir cikariliyor: {len (eksik )} part",flush =True )
        t0 =time .time ()
        for i ,p in enumerate (eksik ,1 ):
            try :
                ob [p ]=brep_snap .exact_cylinders (S [p ])
            except Exception :
                ob [p ]=[]
            if i %100 ==0 :
                print (f"  {i }/{len (eksik )}  {(time .time ()-t0 )/i :.1f}s/part",flush =True )
                with open (yol ,"wb")as f :
                    pickle .dump (ob ,f )
        with open (yol ,"wb")as f :
            pickle .dump (ob ,f )
    return ob 


def veri_kur (rec_ ,model ,ob ,match_greedy ):
    """(feature, label) ciftleri. Etiket: this secenek adayi ROBOT-HAZIR yapar mi."""
    X ,y ,grp =[],[],[]
    for pid ,r in rec_ .items ():
        pak =parca_adaylari (r ,model ,ob )
        if pak is None :
            continue 
        P ,D ,S ,komsu =pak 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        _t ,_f ,_n ,bi =match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )
        cy =ob .get (pid )or []
        for (pi ,gi ,*_x )in bi ["eslesme"]:
            sec =secenekler (cy ,P [pi ],D [pi ],r ["diag"],float (S [pi ]),komsu ,len (P ))
            for (pp ,dd ,ozn )in sec :
                v =pp -G [gi ]
                yan =float (np .linalg .norm (v -(v @Gd [gi ])*Gd [gi ]))
                aci =np .degrees (np .arccos (np .clip (float (dd @Gd [gi ]),-1 ,1 )))
                X .append (ozn );y .append (int (aci <=ROBOT_ACI and yan <=ROBOT_YANAL ))
                grp .append (pid )
    return np .asarray (X ,float ),np .asarray (y ,int ),np .asarray (grp ,str )


def uygula (rec_ ,model ,ob ,sec ,threshold ,match_greedy ,f1w ,mfgler =None ,sayac =None ):
    T ,R =[],[]
    for pid ,r in rec_ .items ():
        if mfgler is not None and r ["mfg"]not in mfgler :
            continue 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="very"if r ["n"]>=8 else "dusuk"
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        pak =parca_adaylari (r ,model ,ob )
        if pak is not None :
            P ,D ,S ,komsu =pak 
            if sec is not None :
                cy =ob .get (pid )or []
                P2 =P .copy ();D2 =D .copy ()
                for i in range (len (P )):
                    opt =secenekler (cy ,P [i ],D [i ],r ["diag"],float (S [i ]),komsu ,len (P ))
                    if len (opt )==1 :
                        continue 
                    Xo =np .asarray ([o [2 ]for o in opt ],float )
                    sk =sec .predict_proba (_uyumlu (sec ,Xo ))[:,1 ]
                    j =int (np .argmax (sk ))
                    # MEVCUT'u only secenek BELIRGIN sekilde onden whereas birak
                    if j !=0 and sk [j ]>=sk [0 ]+threshold :
                        P2 [i ]=opt [j ][0 ];D2 [i ]=opt [j ][1 ]
                        if sayac is not None :
                            sayac ["degisen"]+=1 
                    if sayac is not None :
                        sayac ["candidate"]+=1 
                P ,D =P2 ,D2 
        T .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
        R .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,ROBOT_ACI ,
        False ,signed =True )[:3 ])
    return f1w (T ),f1w (R )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--training-part",type =int ,default =900 )
    ap .add_argument ("--gate",default ="results/wire_gate_v5.pkl")
    a =ap .parse_args ()
    import protocol 
    protocol .tez_dogrula ()
    from sklearn .ensemble import RandomForestClassifier 
    from sina_cluster import match_greedy ,f1w 

    with open (a .gate ,"rb")as f :
        gate =pickle .load (f )

        # --- EGITIM: only training korpusu ureticileri (sinavin 8'i BURADA YOK)
        # KORPUS ORTAM DEGISKENI (2026-08-08 / A5): g10 with yeniden turetilen corpus
        # (`zengin_parite_v4_g10.npz`) with fit edilebilsin. Sabit kalsaydi p3c ESKI
        # dagilimda kalir and gate v7 with UYUMSUZ olurdu ("two gate same dagilimda
        # egitilmeli" dersinin p3c karsiligi).
    _kk =os .environ .get ("P3C_KORPUS","results/zengin_parite_v3.npz")
    print (f"corpus: {_kk }",flush =True )
    d =np .load (_kk ,allow_pickle =True )
    egt_pid =sorted (set (map (str ,d ["pids"])))
    ek =d6_record .yukle (set (egt_pid ))
    # manufacturer-dengeli ornekle
    grup =collections .defaultdict (list )
    for p ,r in ek .items ():
        grup [r ["mfg"]].append (p )
    rng =np .random .RandomState (0 )
    pay =max (1 ,a .egitim_parca //max (len (grup ),1 ))
    sec_pid =[]
    for m ,ps in sorted (grup .items ()):
        ps =sorted (ps )
        sec_pid +=list (rng .choice (ps ,min (pay ,len (ps )),replace =False ))
    ek ={p :ek [p ]for p in sec_pid }
    print (f"EGITIM: {len (ek )} part | manufacturer "
    f"{dict (collections .Counter (r ['mfg']for r in ek .values ()))}")
    ob_e =silindir_onbellek (list (ek ),SILINDIR_EGT )
    X ,y ,grp =veri_kur (ek ,gate ,ob_e ,match_greedy )
    print (f"  training cifti: {len (y )} | pozitif %{100 *y .mean ():.1f} | "
    f"part {len (set (grp ))}")
    sec =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ,class_weight ="balanced")
    sec .fit (X ,y )
    print ("  onem: "+", ".join (f"{n }={v :.3f}"for n ,v in 
    sorted (zip (OZ_AD ,sec .feature_importances_ ),
    key =lambda t :-t [1 ])[:6 ]))
    with open (MODEL ,"wb")as f :
        pickle .dump ({"clf":sec ,"oz":OZ_AD ,"mm_max":MM_MAX },f )

        # --- OLCUM: temiz exam, DEV yarisinda threshold secimi
    sv =d6_record .exam ()
    rec_ =d6_record .yukle (set (sv ["pidler"]))
    with open (SILINDIR_SIN ,"rb")as f :
        ob_s =pickle .load (f )
    dev ={p :r for p ,r in rec_ .items ()if r ["mfg"]in DEV_MFG }
    sin ={p :r for p ,r in rec_ .items ()if r ["mfg"]not in DEV_MFG }
    d0t ,d0r =uygula (dev ,gate ,ob_s ,None ,0 ,match_greedy ,f1w )
    print (f"\nDEV secicisiz: tespit {d0t :.4f} robot {d0r :.4f}")
    en ,en_r ,izgara =None ,d0r ,{}
    for threshold in (0.00 ,0.05 ,0.10 ,0.20 ,0.30 ):
        tf ,rf =uygula (dev ,gate ,ob_s ,sec ,threshold ,match_greedy ,f1w )
        izgara [str (threshold )]={"tespit":tf ,"robot":rf }
        print (f"  threshold {threshold :.2f}: tespit {tf :.4f} robot {rf :.4f}")
        if rf >en_r :
            en_r ,en =rf ,threshold 
    if en is None :
        print ("\nDEV'de hicbir threshold tabani gecmedi -> KOL KAPANDI")
        with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
            json .dump ({"karar":"KAPANDI","dev_taban_robot":d0r ,"izgara":izgara },
            f ,indent =1 )
        return 
    sc =collections .Counter ()
    s0t ,s0r =uygula (sin ,gate ,ob_s ,None ,0 ,match_greedy ,f1w )
    s1t ,s1r =uygula (sin ,gate ,ob_s ,sec ,en ,match_greedy ,f1w ,sayac =sc )
    print (f"\n--- SINAV YARISI (TEK ATIS) ---")
    print (f"{'ayar':<20}{'TESPIT':>9}{'ROBOT':>9}")
    print (f"{'secicisiz':<20}{s0t :>9.4f}{s0r :>9.4f}")
    print (f"{f'selector threshold {en :.2f}':<20}{s1t :>9.4f}{s1r :>9.4f}")
    print (f"{'FARK':<20}{s1t -s0t :>+9.4f}{s1r -s0r :>+9.4f}")
    print (f"  degistirilen candidate: {sc ['degisen']}/{sc ['candidate']}")
    karar ="DAGIT"if s1r >s0r and s1t >=s0t -0.01 else "GERI AL"
    print (f"\nKARAR: {karar }")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"egitim_parca":len (ek ),"egitim_cifti":int (len (y )),
        "izgara":izgara ,"dev_esik":en ,
        "sinav_secicisiz":{"tespit":s0t ,"robot":s0r },
        "sinav_secicili":{"tespit":s1t ,"robot":s1r },
        "degisen":sc ["degisen"],"candidate":sc ["candidate"],"karar":karar },
        f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
