# -*- coding: utf-8 -*-
"""P2: ZENGIN BLOKLAR A/B -- 22 column vs 22+33, AYNI adaylarla.

Denetimin P1 maddesi. Eski evidence (results/rich_gate_receipt.json) 13 SUTUNLU gate zamanindan:
PART_out +0.0351, FAMILY_out_STRICT +0.0489, AUC 0.9275->0.9549 / 0.8956->0.9373.
Gate that gunden beri 22 sutuna output (B-rep fiziksel + icbukey topoloji). Zengin bloklarin
KATTIGI SEY still present mi?

TASARIM: `build_zengin_parite.py` HEM 22 temel HEM 33 zengin sutunu AYNI candidates for uretti.
Yani A/B'de single degisken feature kumesi. (Ayri kosularda uretilmis two npz'yi karsilastirmak
this projede more before yaniltmisti.)

KOLLAR:
  A  22 temel (dagitilan)
  B  22 + konum-9
  C  22 + very-radius-24
  D  22 + 33 (all of them)
TAPER YOK -- olculmus olu, denetim de "ekleme" dedi.

Dagitilan yapinin DONUSUMU (part-ici z-skor) each kolda uygulanir -- urunle same.

KUME: measure_set (split3, dedup, LOCKED disarida). BOOTSTRAP: geometri grubu.
DECISION: karar_olcutu (GA karara katilir).
"""
import collections 
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
ZEN ="results/zengin_parite.npz"


def main ():
    import karar_olcutu 
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    if not os .path .exists (ZEN ):
        raise SystemExit (f"YOK: {ZEN }")
    d =np .load (ZEN ,allow_pickle =True )
    X22 =np .asarray (d ["X22"],float );XR =np .asarray (d ["XR"],float )
    y =np .asarray (d ["y"]);pid =np .array ([str (x )for x in d ["pids"]])
    mfg =np .array ([str (x )for x in d ["mfg"]])
    ad =[str (x )for x in d ["zengin_ad"]]
    print (f"{X22 .shape [0 ]} candidate | temel {X22 .shape [1 ]} | zengin {XR .shape [1 ]} "
    f"({len (ad )} ad) | {len (np .unique (pid ))} part | votes maks {d ['votes'].max ()}")

    KONUM =list (range (0 ,9 ))# A1(3) + A2(6)
    CYARI =list (range (9 ,33 ))# B (24)
    KOL ={"A 22 temel":None ,"B +konum9":KONUM ,"C +cokyaricap24":CYARI ,
    "D +33 hepsi":list (range (XR .shape [1 ]))}

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ()
    measure_set .rapor_bas (rap )
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")
    gk =measure_set .geo_anahtarlari ()
    grp =np .array ([gk .get (p ,"yok:"+p )for p in pid ])
    tg ={r ["geo"]for r in DER }
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    DON =dag .get ("donusum")

    # OLCUM tarafinda zengin sutunlar YOK (DER onbellegi only 22 column tasiyor).
    # Bu yuzden A/B ADAY DUZEYINDE not, ZENGIN NPZ'nin KENDI parcalari on,
    # grup-capraz OOF with is done; after kazanan arm uctan uca dogrulanir.
    from sklearn .model_selection import GroupKFold 

    def olc (M ,maske =None ):
        """Grup-capraz OOF + part-ici goreli threshold -> candidate duzeyi F1 (regime-agirliksiz)."""
        idx =np .arange (len (y ))if maske is None else np .where (maske )[0 ]
        o =np .zeros (len (idx ))
        Ms ,ys ,gs ,ps =M [idx ],y [idx ],grp [idx ],pid [idx ]
        for tr ,te in GroupKFold (n_splits =5 ).split (Ms ,ys ,gs ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (Ms [tr ],ys [tr ]).predict_proba (Ms [te ])[:,1 ]
        ORAN =0.5 ;TABAN =0.25 
        m =np .zeros (len (o ),bool )
        for u in np .unique (ps ):
            i =ps ==u ;v =o [i ]
            m [i ]=(v >=ORAN *max (v .max (),1e-9 ))&(v >=TABAN )
        tp =int ((ys .astype (bool )&m ).sum ());fp =int ((~ys .astype (bool )&m ).sum ())
        fn =int ((ys .astype (bool )&~m ).sum ())
        p_ =tp /max (tp +fp ,1 );r_ =tp /max (tp +fn ,1 )
        return 2 *p_ *r_ /max (p_ +r_ ,1e-9 )

    def matris (sec ):
        X =X22 if sec is None else np .hstack ([X22 ,XR [:,sec ]])
        if not DON :
            return X 
        Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
        for u in np .unique (pid ):
            i =np .where (pid ==u )[0 ]
            Z [i ]=wire_gate .within_part (X [i ],DON )
        return Z 

        # Yeterli GRUBU which is ureticiler (GroupKFold 5 fold ister; WAGO like single-parcali
        # ureticiler bolunemez).
    kodlar =sorted (k for k in set (mfg )if len (set (grp [mfg ==k ]))>=10 )
    print (f"\n{'arm':<18}{'genel':>9}"+"".join (f"{k +'-disi':>12}"for k in kodlar ))
    SON ={}
    for k ,sec in KOL .items ():
        M =matris (sec )
        genel =olc (M )
        satir =f"{k :<18}{genel :>9.4f}"
        SON [k ]={"genel":float (genel )}
        for kk in kodlar :
            v =olc (M ,maske =(mfg ==kk ))
            SON [k ][kk ]=float (v )
            satir +=f"{v :>12.4f}"
        print (satir ,flush =True )

    a =SON ["A 22 temel"]
    print (f"\n{'arm':<18}{'genel fark':>12}"+"".join (f"{k :>10}"for k in kodlar ))
    for k in KOL :
        if k .startswith ("A"):
            continue 
        print (f"{k :<18}{SON [k ]['genel']-a ['genel']:>+12.4f}"
        +"".join (f"{SON [k ][kk ]-a [kk ]:>+10.4f}"for kk in kodlar ))
    en =max ((k for k in KOL if not k .startswith ("A")),key =lambda k :SON [k ]["genel"])
    fark =SON [en ]["genel"]-a ["genel"]
    print (f"\nEN IYI: {en } -> {fark :+.4f}")
    print (f"ESKI KANIT (13 sutun zamani): part-out +0.0351 / family-out +0.0489")
    print (f"KARAR: {'UCTAN UCA DOGRULA'if fark >=0.01 else 'KAZANC KUCUK -- uctan uca degmez'}")
    with io .open ("results/p2_zengin_ab.json","w",encoding ="utf-8")as f :
        json .dump ({"kollar":SON ,"en_iyi":en ,"fark":float (fark )},f ,indent =1 )
    print ("receipt -> results/p2_zengin_ab.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
