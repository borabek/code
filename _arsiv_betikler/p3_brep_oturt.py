# -*- coding: utf-8 -*-
"""P3-POZ-A: B-REP TAM OTURTMA -- angle and yanali AYNI ANDA duzeltmeyi dene.

MERDIVEN (`results/d6_tavan_merdiveni.json`): poz kisitlarinin total bedeli
robot 0.1475 -> 0.4479 arasi, i.e. **+0.3004**. Aci single basina +0.130, lateral +0.088.
Ikisi de same fiziksel nesneden gelir: real deligin EKSENI and AGIZ MERKEZI.
STEP dosyasi bunlari TAM as tasiyor (B-rep silindir yuzu).

ONCEKI DENEMELERIN DERSI -- korumali oturtma SART:
  * [[brep-snap-dead]]      : TANIDIK kumede measured, oradaki poz already iyiydi
  * `r13_yakinlik_eksen`    : YAKINLIK-ONCELIKLI secim gorulmemiste NET -30
    (most yakin silindir most zaman komsu a vida deligi / yuva)
Bu yuzden here oturtma UC kapiyla korunur and all of them TARANIR:
  ACI    : silindir ekseni mevcut yonle `aci_max` inside must be
  MESAFE : mouth noktasi adaydan `mm_max` inside must be
  YARICAP: wire_gate'in R_MIN/R_MAX araligi (tel giren hole boyutu)

SECIM YANLILIGI ENGELI: 8 manufacturer ikiye bolunur; ayar DEV yarisinda secilir,
SINAV yarisinda TEK ATIS olculur.
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

ONBELLEK ="results/_d6_silindirler.pkl"
MAKBUZ ="results/p3_brep_oturt.json"
DEV_MFG ={"SUPU","NIT","S+S","SE"}
ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 


def silindir_onbellek (kayit ,yenile =False ):
    """Parca basina B-rep silindirlerini BIR KEZ cikar, diske yaz."""
    import brep_snap 
    from korpus_kimlik import step_kimlik as SK 
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    ob ={}
    if os .path .exists (ONBELLEK )and not yenile :
        with open (ONBELLEK ,"rb")as f :
            ob =pickle .load (f )
    eksik =[p for p in kayit if p not in ob and p in S ]
    if eksik :
        print (f"silindir cikariliyor: {len (eksik )} part",flush =True )
        t0 =time .time ()
        for i ,p in enumerate (eksik ,1 ):
            try :
                ob [p ]=brep_snap .exact_cylinders (S [p ])
            except Exception :
                ob [p ]=[]
            if i %50 ==0 :
                print (f"  {i }/{len (eksik )}  {(time .time ()-t0 )/i :.1f}s/part",flush =True )
                with open (ONBELLEK ,"wb")as f :
                    pickle .dump (ob ,f )
        with open (ONBELLEK ,"wb")as f :
            pickle .dump (ob ,f )
    n =sum (1 for p in kayit if ob .get (p ))
    print (f"silindir onbellegi: {len (ob )} part | silindiri OLAN {n }/{len (kayit )}")
    return ob 


def oturt (cyls ,P ,D ,aci_max ,mm_max ):
    """Korumali oturtma. Doner: (P2, D2, kac_tanesi_oturdu)."""
    import brep_snap 
    if not cyls or not len (P ):
        return P ,D ,0 
    P2 =P .copy ();D2 =D .copy ();n =0 
    kos =np .cos (np .radians (aci_max ))
    for i in range (len (P )):
        p ,d =P [i ],D [i ]
        en =None 
        for c in cyls :
            if not (brep_snap .R_MIN <=c ["radius"]<=brep_snap .R_MAX ):
                continue 
            eks =c ["axis"]
            if abs (float (eks @d ))<kos :# ACI KAPISI: axis yonle uyusmali
                continue 
            for m in (c ["mouth_a"],c ["mouth_b"]):
                dist =float (np .linalg .norm (m -p ))
                if dist >mm_max :
                    continue 
                if en is None or dist <en [0 ]:
                    en =(dist ,np .asarray (m ,float ),
                    eks if float (eks @d )>=0 else -eks )
        if en is not None :
            P2 [i ]=en [1 ];D2 [i ]=en [2 ];n +=1 
    return P2 ,D2 ,n 


def puanla (kayit ,model ,ob ,aci_max ,mm_max ,match_greedy ,f1w ,ratio =0.40 ,baseline =0.30 ,
mfgler =None ,sayac =None ):
    import wire_gate 
    from p1c_threshold import maske 
    T ,R =[],[]
    for pid ,r in kayit .items ():
        if mfgler is not None and r ["mfg"]not in mfgler :
            continue 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        M =d6_record .x58 (r )
        if M is not None and r .get ("P")is not None and len (r ["P"])and M .shape [1 ]*2 ==model ["n_feat"]:
            k =maske (np .asarray (wire_gate .decision_score (model ,M ),float ),ratio ,baseline )
            if k .any ():
                P =np .asarray (r ["P"],float )[k ];D =np .asarray (r ["Pd"],float )[k ]
        if aci_max is not None and len (P ):
            P ,D ,n =oturt (ob .get (pid )or [],P ,D ,aci_max ,mm_max )
            if sayac is not None :
                sayac ["oturan"]+=n ;sayac ["candidate"]+=len (P )
        T .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
        R .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,ROBOT_ACI ,
        False ,signed =True )[:3 ])
    return f1w (T ),f1w (R )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--model",default ="results/wire_gate_v5.pkl")
    ap .add_argument ("--yenile",action ="store_true")
    a =ap .parse_args ()
    import protocol 
    protocol .tez_dogrula ()
    from sina_cluster import match_greedy ,f1w 

    sv =d6_record .exam ()
    kayit =d6_record .yukle (set (sv ["pidler"]))
    with open (a .model ,"rb")as f :
        model =pickle .load (f )
    ob =silindir_onbellek (kayit ,a .yenile )
    dev ={p :r for p ,r in kayit .items ()if r ["mfg"]in DEV_MFG }
    sin ={p :r for p ,r in kayit .items ()if r ["mfg"]not in DEV_MFG }

    d0t ,d0r =puanla (dev ,model ,ob ,None ,0 ,match_greedy ,f1w )
    print (f"\nDEV oturtmasiz: tespit {d0t :.4f} robot {d0r :.4f}")
    print (f"\n{'aci\\mm':<9}"+"".join (f"{m :>10.0f}"for m in (2 ,4 ,6 ,10 )))
    en_iyi ,en_iyi_r =None ,d0r 
    izgara ={}
    for ac in (10 ,20 ,30 ,45 ):
        satir =[]
        for mm in (2 ,4 ,6 ,10 ):
            tf ,rf =puanla (dev ,model ,ob ,ac ,mm ,match_greedy ,f1w )
            izgara [f"{ac }/{mm }"]={"tespit":tf ,"robot":rf }
            satir .append (rf )
            if rf >en_iyi_r :
                en_iyi_r ,en_iyi =rf ,(ac ,mm )
        print (f"{ac :<9}"+"".join (f"{v :>10.4f}"for v in satir ))

    if en_iyi is None :
        print ("\nDEV'de HICBIR ayar oturtmasizi gecmedi -> KOL KAPANDI")
        with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
            json .dump ({"karar":"KAPANDI","dev_taban_robot":d0r ,"izgara":izgara },
            f ,indent =1 )
        return 
    ac ,mm =en_iyi 
    print (f"\nDEV'de secilen: aci<={ac } derece, mesafe<={mm }mm -> robot {en_iyi_r :.4f} "
    f"(oturtmasiz {d0r :.4f}, {en_iyi_r -d0r :+.4f})")

    sc =collections .Counter ()
    s0t ,s0r =puanla (sin ,model ,ob ,None ,0 ,match_greedy ,f1w )
    s1t ,s1r =puanla (sin ,model ,ob ,ac ,mm ,match_greedy ,f1w ,sayac =sc )
    print (f"\n--- SINAV YARISI (TEK ATIS) ---")
    print (f"{'ayar':<18}{'TESPIT':>9}{'ROBOT':>9}")
    print (f"{'oturtmasiz':<18}{s0t :>9.4f}{s0r :>9.4f}")
    print (f"{f'oturtma {ac }/{mm }':<18}{s1t :>9.4f}{s1r :>9.4f}")
    print (f"{'FARK':<18}{s1t -s0t :>+9.4f}{s1r -s0r :>+9.4f}")
    print (f"  oturan candidate: {sc ['oturan']}/{sc ['candidate']}")
    karar ="DAGIT"if s1r >s0r and s1t >=s0t -0.01 else "GERI AL"
    print (f"\nKARAR: {karar }")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"model":a .model ,"izgara":izgara ,"dev_secim":{"aci":ac ,"mm":mm },
        "dev_taban_robot":d0r ,"sinav_oturtmasiz":{"tespit":s0t ,"robot":s0r },
        "sinav_oturtmali":{"tespit":s1t ,"robot":s1r },
        "oturan":sc ["oturan"],"candidate":sc ["candidate"],"karar":karar },
        f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
