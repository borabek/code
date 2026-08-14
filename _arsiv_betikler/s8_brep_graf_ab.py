# -*- coding: utf-8 -*-
"""S8: B-REP GRAF ozellikleri A/B -- tespit sıralamasi duvarina karsi.

Denetimin P1 maddesi. Simdiye up to B-rep'ten only TEK YUZ okunuyordu (radius, axis);
yuzlerin BIRBIRIYLE ILISKISI never kullanilmadi. Sekiz feature (two tanesi empty ciktigi for
DUSURULDU -- yazilmamislardi, sifir tasiyorlardi):

    g_kom_n / g_kom_cyl / g_kom_pl   mouth cevresindeki B-rep face sayilari
    g_esek_n / _r_ort / _r_yay       eseksenli silindir zinciri (huni/kademe imzasi)
    g_kor / g_gecis                  channel kor mu, boydan boya mi (boydan boya = tel girisi DEGIL)

Kapsama %96.6.

TASARIM: same candidates, single degisken feature kumesi. Once candidate duzeyi (ucuz eleme), gecerse
UCTAN UCA (this arastirmada candidate duzeyi ALTI KEZ yaniltti).

KILL: uctan uca tespit >= +0.01 VE decision_criterion five sarti.
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
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
BOS =("g_agiz_cev","g_yuz_alan")# yazilmadi, hep sifir -> dusuruldu


def main ():
    import decision_criterion 
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    gr =np .load ("results/brep_graf.npz",allow_pickle =True )
    GX =np .asarray (gr ["X"],float );GAD =[str (x )for x in gr ["name"]]
    tut =[i for i ,a in enumerate (GAD )if a not in BOS ]
    GX =GX [:,tut ]
    print (f"graf ozellikleri: {GX .shape } ({len (tut )} sutun, {len (BOS )} bos dusuruldu)")

    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    X58 =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    y =np .asarray (zen ["y"]);pid =np .array ([str (x )for x in zen ["pids"]])
    mfg =np .array ([str (x )for x in zen ["mfg"]])
    assert len (GX )==len (X58 ),(GX .shape ,X58 .shape )

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_zengin_graf.pkl")
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")
    gk =measure_set .geo_anahtarlari ()
    grp =np .array ([gk .get (p ,"absent:"+p )for p in pid ])
    tg ={r ["geo"]for r in DER }
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")

    def donustur (X ):
        if not DON :
            return X 
        Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
        for u in np .unique (pid ):
            i =np .where (pid ==u )[0 ]
            Z [i ]=wire_gate .within_part (X [i ],DON )
        return Z 

        # --- 1) ADAY DUZEYI eleme
    print ("\nADAY DUZEYI (grup-capraz OOF, part-ici goreli threshold):")
    def olc (M ):
        o =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (M ,y ,grp ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (M [tr ],y [tr ]).predict_proba (M [te ])[:,1 ]
        m =np .zeros (len (o ),bool )
        for u in np .unique (pid ):
            i =pid ==u ;v =o [i ]
            m [i ]=(v >=0.5 *max (v .max (),1e-9 ))&(v >=0.25 )
        tp =int ((y .astype (bool )&m ).sum ());fp =int ((~y .astype (bool )&m ).sum ())
        fn =int ((y .astype (bool )&~m ).sum ())
        p_ =tp /max (tp +fp ,1 );r_ =tp /max (tp +fn ,1 )
        return 2 *p_ *r_ /max (p_ +r_ ,1e-9 )
    a58 =olc (donustur (X58 ));a68 =olc (donustur (np .hstack ([X58 ,GX ])))
    print (f"  58 sutun {a58 :.4f} | 58+graf {a68 :.4f} | fark {a68 -a58 :+.4f}")

    # --- 2) UCTAN UCA (karar here)
    gr_der ={}
    for i ,p in enumerate (pid ):
        gr_der .setdefault (p ,[]).append (i )
    kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in pid [mfg ==k ]).most_common (1 )[0 ][0 ]
    for k in np .unique (mfg )}
    BOLME =[("tanidik",None ,DER )]
    for k ,mad in kod .items ():
        alt =[x for x in DER if x ["mfg"]==mad ]
        if len (alt )>=10 :
            BOLME .append ((mad ,k ,alt ))

            # measurement tarafinda graf sutunlari: DER parcalarinin satirlarindan al
    def der_graf (r ):
        """Graf sutunlari ADAYIN KENDI noktasindan hesaplandi (`_der_zengin_graf.pkl`).

        Ilk surumde zengin npz'den PID with eslestiriliyordu and 34/194 parcada candidate count
        tutmuyordu -> that parts SIFIR CP aliyor and metrik cokuyordu. Ozellik not OLCUM
        hatasiydi."""
        g =r .get ("GX")
        return g [:,tut ]if g is not None else None 

    eksik =sum (1 for r in DER if der_graf (r )is None )
    print (f"\nolcum tarafinda graf sutunu EKSIK olan part: {eksik }/{len (DER )}")

    SON ,PARCA ={},{}
    print (f"\n{'arm':<16}"+"".join (f"{b :>12}"for b ,_ ,_ in BOLME ))
    for ad ,ek in (("A 58",False ),("B 58+graf",True )):
        Xt =np .hstack ([X58 ,GX ])if ek else X58 
        Mt =donustur (Xt )
        SON [ad ]={};sat =f"{ad :<16}"
        for b ,mk ,alt in BOLME :
            keep =~np .isin (grp ,list (tg ))
            if mk is not None :
                keep =keep &(mfg !=mk )
            clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (Mt [keep ],y [keep ])
            m ={"clf":clf ,"n_feat":Mt .shape [1 ],"donusum":DON }
            det ,rob =[],[]
            for r in alt :
                P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
                Xr =np .hstack ([r ["X"],r ["XR"]])if r .get ("XR")is not None else None 
                if Xr is not None and ek :
                    G_ =der_graf (r )
                    # Graf sutunu otherwise NOTR (sifir) -- parcayi DUSURME (first surumun hatasi)
                    Xr =np .hstack ([Xr ,G_ if G_ is not None 
                    else np .zeros ((len (Xr ),len (tut )))])
                if Xr is not None :
                    s =wire_gate .decision_score (m ,Xr )
                    k2 =wire_gate .decision_mask (s )
                    if k2 .any ():
                        P =r ["P"][k2 ].copy ();Pd =r ["Pd"][k2 ].copy ()
                rj ="very"if r ["n"]>=8 else "low"
                det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
                rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
            SON [ad ][b ]=float (f1w (det ))
            PARCA [(ad ,b )]=(det ,rob ,[x ["geo"]for x in alt ])
            sat +=f"{f1w (det ):>12.4f}"
        print (sat ,flush =True )

    ga ={}
    for b ,_ ,_ in BOLME :
        if b =="tanidik":
            continue 
        da ,_ ,g =PARCA [("A 58",b )];db ,_ ,_ =PARCA [("B 58+graf",b )]
        fn =lambda rows :f1w ([y2 for _ ,y2 in rows ])-f1w ([x for x ,_ in rows ])
        _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (da ,db )),g ,fn ,n =2000 )
        ga [b ]=(lo ,hi )
    print ("\n=== DECISION ===")
    k =decision_criterion .degerlendir (SON ["A 58"],SON ["B 58+graf"],ga =ga )
    print (k )
    with io .open ("results/s8_brep_graf_ab.json","w",encoding ="utf-8")as f :
        json .dump ({"aday_58":float (a58 ),"aday_68":float (a68 ),
        "kollar":SON ,"gecti":bool (k )},f ,indent =1 )
    print ("receipt -> results/s8_brep_graf_ab.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
