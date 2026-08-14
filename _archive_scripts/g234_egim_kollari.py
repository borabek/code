# -*- coding: utf-8 -*-
"""G2/G3/G4: EGIM KOLLARI -- same veriden more very GENELLEME cikar.

G1 ceiling hesabi ([[candidate-absent-temsil-darbogazi]]): gate MUKEMMEL olsa (sifir FP dahil) bile
unseen-manufacturer tavani **0.5964**. Yani this three arm 0.70'i TEK BASINA GETIREMEZ; amaclari
that tavanin under ek pay almak. Asil arm G5 (temsil).

UC KOL, UCU DE AYRI MODEL (birlestirilmez -- otherwise hangisinin ne getirdigi atfedilemez):
  G2 URETICI-DENGELI : training satirlarinin %69'u WEI+PXC. Uretici basina equal total weight.
  G3 IKIZ-AGIRLIKLI  : geometri grubu basina ~2.5 kopya; each GRUP equal total weight.
  G4 BUDAMA          : 13 ozniteligin only `votes`'u transfer ediyor
                       ([[gate-memorizes-not-learns]]). Dusuk kapasite + part-ici z-score
                       sutunlari (mutlak buyuklukler ureticiye ozgudur, goreli olanlar not).

HER KOL IKI KUMEDE RAPORLANIR -- 194'luk cluster and unseen-manufacturer sinavi. Yalniz birinde
bakmak yaniltir: [[unseen-manufacturer-sinavi-first-measurement]] data kolunda +0.0058 vs +0.0535
gostermisti.

G4 TUZAGI: dejenere "hepsini kabul et" modeli de mukemmel transfer eder. Bu yuzden
KESINLIK TABANI sart -- precision 0.60'in altina duserse arm GECMEZ.

TEZ DEGISMEZ: network egitilmez, remesh does not change, `v_o` does not change. Yalniz gate'in EGITIM
AGIRLIKLARI and OZNITELIK ALT KUMESI degisir.
"""
import collections 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import wire_gate 
from f2_12_data_kolu import olc 

V3 ="results/zengin_parite_v3.npz"
SINAV ="results/_der_sinav_yeni.pkl"
MAKBUZ ="results/g234_egim_kollari.json"


def _z (X ,pid ):
    """Parca-ici z-score -- calisma anindaki with AYNI (part bazinda)."""
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ),float )
    for p in np .unique (pid ):
        m =pid ==p 
        Z [m ]=wire_gate .within_part (X [m ],"zskor")
    return Z 


def egit (npz ,wgt_ =None ,cols =None ,depth =None ,yaprak =3 ,seed =0 ):
    """weight: None | 'manufacturer' | 'grup'  ·  cols: kullanilacak HAM column indeksleri."""
    from sklearn .ensemble import RandomForestClassifier 
    d =np .load (npz ,allow_pickle =True )
    X =np .hstack ([d ["X22"],d ["XR"]]).astype (float )
    pid =np .asarray (d ["pids"],str );mfg =np .asarray (d ["mfg"],str )
    Z =_z (X ,pid )
    # SUTUN SECIMI KALDIRILDI -- URUNUN DECISION YOLUYLA UYUMSUZDU (2026-08-04):
    # `wire_gate.decision_score` before TUM 58 sutuna part-ici z-score uygular (->116), after
    # `Xd[:, :n_feat]` with ONEK takes. Yani "13 ham + onlarin z'si" like a SECIM cikarimda
    # okunamaz; model 26 sutunla egitilir but cikarimda ham 0..25 okunur -> F1 TAM 0.0000.
    # (Ilk kosuda aynen this became; sonuc sanilmasin diye kayda geciyor.)
    # Ayni hipotez -- "gate EZBERLIYOR, kapasite dusurulunce transfer artar" -- column
    # budamadan da sinanabilir: KAPASITE kisitlamasi (depth/yaprak). O path urunun
    # karar yoluyla TAM UYUMLU because feature duzeni degismiyor.
    w =None 
    if wgt_ =="manufacturer":
        c =collections .Counter (mfg .tolist ())
        w =np .array ([1.0 /c [m ]for m in mfg ]);w *=len (w )/w .sum ()
    elif wgt_ =="grup":
        import measure_set as OK 
        gk =OK .geo_anahtarlari ()
        g =np .array ([gk .get (p ,"absent:"+p )for p in pid ])
        c =collections .Counter (g .tolist ())
        w =np .array ([1.0 /c [x ]for x in g ]);w *=len (w )/w .sum ()
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =yaprak ,
    max_depth =depth ,n_jobs =-1 ,random_state =seed )
    clf .fit (Z ,d ["y"],sample_weight =w )
    return {"clf":clf ,"n_feat":Z .shape [1 ],"donusum":"zskor",
    "cols":(list (cols )if cols is not None else None ),"feat_names":None }


def main ():
    import protocol 
    protocol .tez_dogrula ()
    import measure_set as OK 
    D194 ,_ =OK .cluster ("results/_der_tam.pkl")
    with open (SINAV ,"rb")as f :
        DSIN =pickle .load (f )
    print (f"measurement: 194'luk {len (D194 )} part | exam {len (DSIN )} part\n")

    # G4 column secimi: `votes` (11) + part-ici z-skoru transfer eden single sey oldugu for
    # baseline 13 sutunun GORELI hali. Mutlak size/depth sutunlari ureticiye ozgudur.
    TABAN13 =list (range (13 ))

    KOL =[
    ("v3 TABAN (degismemis)",dict ()),
    ("G2 manufacturer-dengeli",dict (wgt_ ="manufacturer")),
    ("G3 ikiz-agirlikli",dict (wgt_ ="grup")),
    ("G4a kapasite low (depth 8, yaprak 10)",dict (depth =8 ,yaprak =10 )),
    ("G4b kapasite very low (depth 5, yaprak 25)",dict (depth =5 ,yaprak =25 )),
    ]
    S ={}
    print (f"{'arm':<38}{'194 F1':>9}{'SINAV F1':>10}{'exam precision':>16}")
    for ad ,kw in KOL :
        m =egit (V3 ,**kw )
        r1 =olc (m ,D194 );r2 =olc (m ,DSIN )
        kes =r2 ["TP"]/max (r2 ["TP"]+r2 ["FP"],1 )
        S [ad ]={"m194":r1 ["F1"],"exam":r2 ["F1"],"precision":kes ,
        "TP":r2 ["TP"],"FP":r2 ["FP"],"FN":r2 ["FN"]}
        print (f"{ad :<38}{r1 ['F1']:>9.4f}{r2 ['F1']:>10.4f}{kes :>16.4f}")

    t =S ["v3 TABAN (degismemis)"]
    print (f"\n{'arm':<38}{'194 difference':>10}{'SINAV difference':>12}   GO")
    GO ={"G2 manufacturer-dengeli":(0.020 ,-0.010 ),"G3 ikiz-agirlikli":(0.015 ,-1.0 ),
    "G4a kapasite low (depth 8, yaprak 10)":(0.025 ,-1.0 ),
    "G4b kapasite very low (depth 5, yaprak 25)":(0.025 ,-1.0 )}
    for ad ,(esik_s ,esik_p )in GO .items ():
        d1 =S [ad ]["m194"]-t ["m194"];d2 =S [ad ]["exam"]-t ["exam"]
        ok =d2 >=esik_s and d1 >=esik_p 
        if ad .startswith ("G4"):
            ok =ok and S [ad ]["precision"]>=0.60 
        print (f"{ad :<38}{d1 :>+10.4f}{d2 :>+12.4f}   {'GECTI'if ok else 'gecmedi'}")

    en =max (S ,key =lambda k :S [k ]["exam"])
    print (f"\nEN IYI SINAV F1: {en } -> {S [en ]['exam']:.4f}")
    print (f"GUN 1 KAPISI (>= 0.47): {'GECTI'if S [en ]['exam']>=0.47 else 'GECMEDI'}")
    print (f"  NOT: G1 ceiling hesabi already 0.5964 diyordu -- this kollar 0.70'i tek basina")
    print (f"       getiremez. Asil arm G5 (temsil), ADAY_YOK 563 FN kovasi.")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump (S ,f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
