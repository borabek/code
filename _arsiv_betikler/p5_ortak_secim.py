# -*- coding: utf-8 -*-
"""P5: GATE VE POZ BIRLIKTE SECILIR -- mevcut `gate -> poz` order wrong may be.

HIPOTEZ: bugunku boru hatti before gate'i uygular, after hayatta kalanlarin pozunu fixes.
Dogru poza sahip a candidate gate'te ERKENDEN elenirse that CP a more geri gelmez. Oysa gate
adayi LOOSE tespit etiketiyle (lateral <= max(3mm,%6), angle serbest) egitilmis; robot-hazir
olup olmayacagini never sormamis.

BU KOL: each (candidate x poz) ciftini DOGRUDAN **robot-hazir** etiketiyle skorlar
(lateral <= 2mm VE signed angle <= 10). Yani selector, "this candidate gecer mi" instead of
"this candidate BU POZLA robotun kullanabilecegi a CP gives mi" sorusunu ogrenir.

DISIPLIN:
  * HAM pool is used (gate ONCESI) -- kolun butun anlami this
  * manufacturer kimligi OZNITELIK DEGIL
  * URETICI-DISI dis capraz gecerleme; threshold IC katmanda secilir
  * CAKISMA COZUMU: a candidate only BIR times is used (Macar atama)
  * confidence dusukse MEVCUT poz korunur (geri donus)

TEZ DEGISMEZ: `v_o` turetmesi, 5 sinif, ~6000 remesh aynen. Bu a SECIM katmanidir.
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

MAKBUZ ="results/p5_ortak_secim.json"
MODEL ="results/p5_ortak_secim.pkl"
DEV_MFG ={"SUPU","NIT","S+S","SE"}
ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 
MM =8.0 


def poz_secenekleri (cyl ,ac ,p ,d ,diag ):
    """MEVCUT + silindir agizlari + duzlemsel opening merkezleri (+- direction)."""
    import brep_snap 
    out =[(np .asarray (p ,float ),np .asarray (d ,float ),1.0 ,0.0 ,0.0 )]
    for c in (cyl or []):
        if not (brep_snap .R_MIN <=c ["radius"]<=brep_snap .R_MAX ):
            continue 
        for m in (c ["mouth_a"],c ["mouth_b"]):
            mm =float (np .linalg .norm (np .asarray (m ,float )-p ))
            if mm >MM :
                continue 
            a =np .asarray (c ["axis"],float )
            for s in (1.0 ,-1.0 ):
                out .append ((np .asarray (m ,float ),s *a ,0.0 ,mm ,float (c ["radius"])))
    for o in (ac or []):
        m =np .asarray (o ["center"],float )
        mm =float (np .linalg .norm (m -p ))
        if mm >MM :
            continue 
        n =np .asarray (o ["normal"],float )
        for s in (1.0 ,-1.0 ):
            out .append ((m ,s *n ,0.0 ,mm ,float (o ["esd_r"])))
    return out 


def cift_ozellik (X58_satir ,pp ,dd ,mevcut ,dist_ ,yaricap ,p0 ,d0 ,diag ,n_aday ):
    """Gate ozellikleri (58) + POZ ozellikleri. Uretici kimligi YOK."""
    aci =float (np .degrees (np .arccos (np .clip (abs (float (dd @d0 )),-1 ,1 ))))
    return np .concatenate ([X58_satir ,[mevcut ,dist_ ,dist_ /max (diag ,1e-6 ),
    yaricap ,aci ,float (n_aday ),diag ]])


def parca_ciftleri (r ,cyl ,ac ):
    """HAM havuzun tum (candidate, poz) ciftleri + ozellikleri."""
    X58 =d6_record .x58 (r )
    P0 =np .asarray (r ["P"],float )if r .get ("P")is not None else np .zeros ((0 ,3 ))
    D0 =np .asarray (r ["Pd"],float )if r .get ("Pd")is not None else np .zeros ((0 ,3 ))
    if X58 is None or not len (P0 ):
        return None 
    Z ,meta =[],[]
    for i in range (len (P0 )):
        for (pp ,dd ,mv ,ms ,yr )in poz_secenekleri (cyl ,ac ,P0 [i ],D0 [i ],r ["diag"]):
            Z .append (cift_ozellik (X58 [i ],pp ,dd ,mv ,ms ,yr ,P0 [i ],D0 [i ],
            r ["diag"],len (P0 )))
            meta .append ((i ,pp ,dd ))
    return np .asarray (Z ,float ),meta 


def etiketle (meta ,G ,Gd ):
    """Robot-hazir mi: lateral <= 2mm VE signed angle <= 10."""
    y =np .zeros (len (meta ),int )
    if not len (G ):
        return y 
    for k ,(_i ,pp ,dd )in enumerate (meta ):
        v =pp -G 
        yan =np .linalg .norm (v -(v *Gd ).sum (1 )[:,None ]*Gd ,axis =1 )
        aci =np .degrees (np .arccos (np .clip (Gd @dd ,-1 ,1 )))
        if np .any ((yan <=ROBOT_YANAL )&(aci <=ROBOT_ACI )):
            y [k ]=1 
    return y 


def sec_ve_uygula (Z ,meta ,clf ,threshold ,P0 ,D0 ):
    """Skorla, candidate basina EN IYI pozu al, esigin altini MEVCUT poza dondur."""
    s =clf .predict_proba (Z )[:,1 ]
    en ={}
    for k ,(i ,pp ,dd )in enumerate (meta ):
        if i not in en or s [k ]>en [i ][0 ]:
            en [i ]=(s [k ],pp ,dd )
    tut ,P ,D =[],[],[]
    for i ,(sk ,pp ,dd )in sorted (en .items ()):
        if sk <threshold :
            continue 
        tut .append (i );P .append (pp );D .append (dd )
    if not P :
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    return np .asarray (P ,float ),np .asarray (D ,float )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--training-part",type =int ,default =600 )
    a =ap .parse_args ()
    import protocol 
    protocol .tez_dogrula ()
    from sklearn .ensemble import RandomForestClassifier 
    from sina_cluster import match_hungarian ,f1w 

    with open ("results/_p3c_silindir_egitim.pkl","rb")as f :
        cyl_e =pickle .load (f )
    with open ("results/_d6_silindirler.pkl","rb")as f :
        cyl_d6 =pickle .load (f )
    ac_d6 ={}
    for f in glob .glob ("results/_d6_acikliklar_*.pkl"):
        with open (f ,"rb")as h :
            ac_d6 .update (pickle .load (h ))

            # EGITIM: training korpusu ureticileri
    d =np .load ("results/zengin_parite_v3.npz",allow_pickle =True )
    ek =d6_record .yukle (set (map (str ,d ["pids"])))
    grup =collections .defaultdict (list )
    for p ,r in ek .items ():
        grup [r ["mfg"]].append (p )
    rng =np .random .RandomState (0 )
    pay =max (1 ,a .egitim_parca //max (len (grup ),1 ))
    sec_pid =[]
    for m ,ps in sorted (grup .items ()):
        ps =sorted (ps )
        sec_pid +=list (rng .choice (ps ,min (pay ,len (ps )),replace =False ))
    print (f"EGITIM: {len (sec_pid )} part")
    X ,y ,mf =[],[],[]
    for pid in sec_pid :
        r =ek [pid ]
        out =parca_ciftleri (r ,cyl_e .get (pid ),None )
        if out is None :
            continue 
        Z ,meta =out 
        yy =etiketle (meta ,np .asarray (r ["G"],float ),np .asarray (r ["Gd"],float ))
        X .append (Z );y .append (yy );mf +=[r ["mfg"]]*len (yy )
    X =np .vstack (X );y =np .concatenate (y );mf =np .asarray (mf )
    print (f"  cift: {len (y )} | robot-hazir %{100 *y .mean ():.2f} | oznitelik {X .shape [1 ]}")

    # URETICI-DISI CV
    from sklearn .metrics import roc_auc_score 
    urs =[m for m ,n in collections .Counter (mf ).most_common ()if n >=2000 ]
    print (f"\nURETICI-DISI CV: {urs }")
    for m in urs :
        tr =mf !=m 
        c =RandomForestClassifier (n_estimators =200 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ,class_weight ="balanced").fit (X [tr ],y [tr ])
        s =c .predict_proba (X [~tr ])[:,1 ]
        au =roc_auc_score (y [~tr ],s )if len (set (y [~tr ]))>1 else float ("nan")
        print (f"  {m :<8} disarida AUC {au :.4f}")
    clf =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ,class_weight ="balanced").fit (X ,y )
    with open (MODEL ,"wb")as f :
        pickle .dump ({"clf":clf },f )

        # OLCUM: D6, DEV yarisinda threshold
    sv =d6_record .exam ()
    rec_ =d6_record .yukle (set (sv ["pidler"]))
    def olc (alt ,threshold ):
        T ,R =[],[]
        for pid ,r in rec_ .items ():
            if r ["mfg"]not in alt :
                continue 
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="very"if r ["n"]>=8 else "low"
            P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
            out =parca_ciftleri (r ,cyl_d6 .get (pid ),ac_d6 .get (pid ))
            if out is not None :
                Z ,meta =out 
                P ,D =sec_ve_uygula (Z ,meta ,clf ,threshold ,None ,None )
            T .append ((rj ,)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
            R .append ((rj ,)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,ROBOT_ACI ,
            False ,signed =True )[:3 ])
        return f1w (T ),f1w (R )
    tum ={r ["mfg"]for r in rec_ .values ()}
    sin_mfg =tum -DEV_MFG 
    print (f"\n{'threshold':<8}{'DEV tespit':>12}{'DEV robot':>11}")
    en ,en_r ,izg =None ,-1.0 ,{}
    for e in (0.30 ,0.40 ,0.50 ,0.60 ,0.70 ):
        tf ,rf =olc (DEV_MFG ,e )
        izg [str (e )]={"tespit":tf ,"robot":rf }
        print (f"{e :<8.2f}{tf :>12.4f}{rf :>11.4f}")
        if rf >en_r :
            en_r ,en =rf ,e 
    st ,sr =olc (sin_mfg ,en )
    print (f"\n--- SINAV YARISI (TEK ATIS, threshold {en :.2f}) ---")
    print (f"  tespit {st :.4f} | robot {sr :.4f}")
    print (f"  KIYAS (mevcut yigin, ayni yari): tespit 0.5394 | robot 0.2676")
    karar ="DAGIT"if sr >0.2676 else "GERI AL"
    print (f"\nKARAR: {karar }")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"izgara":izg ,"dev_esik":en ,"exam":{"tespit":st ,"robot":sr },
        "kiyas_robot":0.2676 ,"karar":karar },f ,indent =1 )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
