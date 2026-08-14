# -*- coding: utf-8 -*-
"""P1: PARITE korpusu with ESKI corpus -- gate hangisiyle more iyi?

2026-08-01 denetimi: gate training verisi adaylari AYRI a yoldan uretiyordu (step_path absent,
union_all). Kod duzeltildi (`robot_cp.derive_candidates` single source) and corpus YENIDEN URETILDI.
Bu betik two training verisini AYNI protokolde karsilastirir:

    ESKI   results/gate_regrow_data_topo.npz      (parite KIRIK: votes 12'ye up to)
    YENI   results/gate_regrow_data_parite.npz    (urunun own candidate ureticisi)

Beklenti a "kazanc" DEGIL, TUTARLILIK: gate residual gercekte gordugu adaylarla egitiliyor.
Ama beklenti with measurement ayri seyler -- karar `karar_olcutu` with, GA'li.

KUME: `measure_set` (split3, dedup, LOCKED disarida). BOOTSTRAP: geometri grubu.
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
ESKI ="results/gate_regrow_data_topo.npz"
YENI ="results/gate_regrow_data_parite.npz"


def main ():
    import karar_olcutu 
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    for f in (ESKI ,YENI ):
        if not os .path .exists (f ):
            raise SystemExit (f"YOK: {f }")
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ()
    measure_set .rapor_bas (rap )
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")
    gk =measure_set .geo_anahtarlari ()
    tg ={r ["geo"]for r in DER }
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    DON =dag .get ("donusum")# dagitilan yapiyi KORU (this an "zskor")

    def data_ (yol ):
        d =np .load (yol ,allow_pickle =True )
        pid =np .array ([str (x )for x in d ["pids"]])
        grp =np .array ([gk .get (p ,"yok:"+p )for p in pid ])
        mfg =np .array ([str (x )for x in d ["mfg"]])
        X =np .asarray (d ["X"],float );y =np .asarray (d ["y"])
        v =np .asarray (d ["votes"])if "votes"in d .files else X [:,11 ]
        Z =None 
        if DON :
            Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
            for u in np .unique (pid ):
                i =np .where (pid ==u )[0 ]
                Z [i ]=wire_gate .within_part (X [i ],DON )
        return dict (pid =pid ,grp =grp ,mfg =mfg ,X =X ,y =y ,Z =Z ,votes =v ,
        n_parca =len (np .unique (pid )))

    E ,Y =data_ (ESKI ),data_ (YENI )
    print (f"\n{'veri':<8}{'candidate':>8}{'part':>8}{'votes maks':>12}{'pozitif':>9}")
    for ad ,D_ in (("ESKI",E ),("YENI",Y )):
        print (f"{ad :<8}{len (D_ ['y']):>8}{D_ ['n_parca']:>8}{D_ ['votes'].max ():>12.0f}"
        f"{D_ ['y'].mean ():>9.3f}")

    kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in E ["pid"][E ["mfg"]==k ]).most_common (1 )[0 ][0 ]
    for k in np .unique (E ["mfg"])}
    BOLME =[("tanidik",None ,DER )]
    for k ,mad in kod .items ():
        alt =[x for x in DER if x ["mfg"]==mad ]
        if len (alt )>=10 :
            BOLME .append ((mad ,k ,alt ))

    def puanla (D_ ,mfg_kod ,alt ):
        keep =~np .isin (D_ ["grp"],list (tg ))
        if mfg_kod is not None :
            keep =keep &(D_ ["mfg"]!=mfg_kod )
        M =D_ ["Z"]if D_ ["Z"]is not None else D_ ["X"]
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (M [keep ],D_ ["y"][keep ])
        m ={"clf":clf ,"n_feat":M .shape [1 ],"donusum":DON }
        det ,rob =[],[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None :
                s =wire_gate .decision_score (m ,r ["X"])
                k =wire_gate .decision_mask (s )
                if k .any ():
                    P =r ["P"][k ];Pd =r ["Pd"][k ]
            rj ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    SON ,PARCA ={},{}
    print (f"\n{'veri':<8}"+"".join (f"{b :>12}"for b ,_ ,_ in BOLME )+f"{'robot':>10}")
    for ad ,D_ in (("ESKI",E ),("YENI",Y )):
        SON [ad ]={};sat =f"{ad :<8}"
        for b ,mk ,alt in BOLME :
            det ,rob =puanla (D_ ,mk ,alt )
            SON [ad ][b ]=float (f1w (det ))
            PARCA [(ad ,b )]=(det ,rob ,[x ["geo"]for x in alt ])
            sat +=f"{f1w (det ):>12.4f}"
        sat +=f"{f1w (PARCA [(ad ,'tanidik')][1 ]):>10.4f}"
        print (sat ,flush =True )

    ga ={}
    for b ,_ ,_ in BOLME :
        if b =="tanidik":
            continue 
        da ,_ ,g =PARCA [("ESKI",b )];db ,_ ,_ =PARCA [("YENI",b )]
        cift =list (zip (da ,db ))
        fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
        _ ,lo ,hi =measure_set .grup_bootstrap (cift ,g ,fn ,n =2000 )
        ga [b ]=(lo ,hi )

    print ("\n=== KARAR (ESKI -> YENI) ===")
    k =karar_olcutu .degerlendir (SON ["ESKI"],SON ["YENI"],ga =ga )
    print (k )
    print (f"\nNOT: parite duzeltmesi bir KAZANC vaadi degil, TUTARLILIK duzeltmesidir. Kural")
    print (f"gecmese bile YENI veri dogru olandir; kural burada 'zarar var mi' sorusunu yanitlar.")
    zarar =(SON ["YENI"]["tanidik"]-SON ["ESKI"]["tanidik"]<-0.02 )or any (
    SON ["YENI"][b ]-SON ["ESKI"][b ]<-0.03 for b ,_ ,_ in BOLME if b !="tanidik")
    print (f"ZARAR VAR MI: {'EVET -- dagitma'if zarar else 'HAYIR -- parite verisi dagitilabilir'}")
    with io .open ("results/p1_parite_karsilastir.json","w",encoding ="utf-8")as f :
        json .dump ({"kollar":SON ,"ga":{a :list (b )for a ,b in ga .items ()},
        "karar":bool (k ),"zarar":bool (zarar ),
        "votes_maks":{"eski":float (E ["votes"].max ()),
        "yeni":float (Y ["votes"].max ())}},f ,indent =1 )
    print ("receipt -> results/p1_parite_karsilastir.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
