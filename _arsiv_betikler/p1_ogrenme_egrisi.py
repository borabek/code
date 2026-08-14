# -*- coding: utf-8 -*-
"""P1: GATE KORPUSU OGRENME EGRISI -- "kac part more is required?" sorusunun SAYISAL cevabi.

WHY SIMDI: gece along nine mekanizma same cepheye carpti and geriye TEK canli arm
kaldi: UCUNCU URETICI VERISI. Ama "more very data" a plan degildir; plan, KAC part
gerektigini bilmektir. Bunu however ogrenme egrisi soyler.

W2 kanitladi ki data real a kaldiractir: +110 new WEI parcasi gorulmemis parcada
+0.0391 verdi (GA sifiri disliyor). Ama egri duz mu, logaritmik mi, doymus mu -- bilmiyoruz.

YONTEM: training korpusunu GEOMETRI GRUBU bazinda lower-orneklle (part not grup: ikizler
birlikte gider, otherwise "new data" yanilsamasi becomes). Her boyutta 3 seed. Olcum kumesi
SABIT kalir. Uc bolmede birden raporlanir.

Sonra log-egri uydurulur and EKSTRAPOLASYON verilir: hedefe ulasmak for gereken corpus
buyuklugu. Ekstrapolasyon DURUSTCE "egri doymuyorsa gecerlidir" serhiyle raporlanir.

Hicbir sey dagitilmaz; this a OLCUMDUR.
"""
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

PAYLAR =[0.125 ,0.25 ,0.40 ,0.60 ,0.80 ,1.00 ]
TOHUMLAR =(0 ,1 ,2 )


def main ():
    import gate_bench as T 
    import measure_set 
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 
    from gece_kilit import guard 

    guard ("p1")
    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    X ,y ,pid ,mfg ,keep =D ["X"],D ["y"],D ["pid"],D ["mfg"],D ["keep"]
    gk =D ["gk"]
    grup =np .array ([gk .get (p ,"yok:"+p )for p in pid ])
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],D ["donusum"])

    def kol_yap (pay ,tohum_alt ):
        def arm (X_ ,y_ ,pid_ ,mfg_ ,kp ,th ):
        # GRUP bazinda lower-ornekleme: ikizler AYNI tarafta kalir
            g_var =np .unique (grup [kp ])
            rng =np .random .default_rng (1000 *tohum_alt +int (pay *1000 ))
            sec =rng .permutation (g_var )[:max (2 ,int (round (pay *len (g_var ))))]
            m =kp &np .isin (grup ,sec )
            if len (np .unique (y_ [m ]))<2 :
                m =kp 
            clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =th ).fit (Z [m ],y_ [m ])
            return {"clf":clf ,"n_feat":Z .shape [1 ],"donusum":D ["donusum"],"_n":int (m .sum ()),
            "_g":len (sec )}
        return arm 

    T .baslik (D )
    EGRI ={}
    for pay in PAYLAR :
        satir =[]
        for th in TOHUMLAR :
            S ,_ =T .calistir (D ,kol_yap (pay ,th ),f"pay={pay :.3f} t{th }",
            tohumlar =(th ,),ayrinti =False )
            satir .append (S )
        hav =float (np .mean ([s ["havuzlanmis"]["tespit"]for s in satir ]))
        ud =float (np .mean ([s ["_URETICI_DISI_ORT"]for s in satir ]))
        rob =float (np .mean ([s ["havuzlanmis"]["robot"]for s in satir ]))
        sd =float (np .std ([s ["havuzlanmis"]["tespit"]for s in satir ]))
        g_var =np .unique (grup [keep ])
        n_g =max (2 ,int (round (pay *len (g_var ))))
        EGRI [pay ]={"grup":n_g ,"havuzlanmis":hav ,"deviation":sd ,
        "uretici_ort":ud ,"robot":rob }
        print (f"pay {pay :<6.3f} grup {n_g :>4}  pool {hav :.4f} (+-{sd :.4f})  "
        f"manufacturer-ort {ud :.4f}  robot {rob :.4f}",flush =True )
        guard (f"p1 pay={pay }")

        # --- LOG EGRI UYDUR: F1 = a + b*ln(grup)
    g =np .array ([EGRI [p ]["grup"]for p in PAYLAR ],float )
    f =np .array ([EGRI [p ]["havuzlanmis"]for p in PAYLAR ],float )
    u =np .array ([EGRI [p ]["uretici_ort"]for p in PAYLAR ],float )
    A =np .vstack ([np .ones_like (g ),np .log (g )]).T 
    kh ,*_ =np .linalg .lstsq (A ,f ,rcond =None )
    ku ,*_ =np .linalg .lstsq (A ,u ,rcond =None )
    print (f"\nLOG UYUM  havuzlanmis: F1 = {kh [0 ]:.4f} + {kh [1 ]:.4f} * ln(grup)")
    print (f"          manufacturer-disi: F1 = {ku [0 ]:.4f} + {ku [1 ]:.4f} * ln(grup)")
    # last two noktanin egimi: DOYUM present mi
    egim_son =(f [-1 ]-f [-2 ])/max (np .log (g [-1 ])-np .log (g [-2 ]),1e-9 )
    print (f"  son iki noktanin egimi {egim_son :+.4f} (tum egri {kh [1 ]:+.4f})"
    f"  -> {'DOYUYOR'if egim_son <kh [1 ]*0.5 else 'DUZ DEVAM'}")

    print (f"\nEKSTRAPOLASYON (yalniz egri doymuyorsa gecerli):")
    simdi_g =int (g [-1 ])
    for hedef in (0.80 ,0.85 ):
        if kh [1 ]>1e-6 :
            gerek =float (np .exp ((hedef -kh [0 ])/kh [1 ]))
            print (f"  havuzlanmis {hedef :.2f} icin ~{gerek :,.0f} geometri grubu "
            f"({gerek /simdi_g :.1f}x, +{gerek -simdi_g :,.0f} grup)")
        else :
            print (f"  havuzlanmis {hedef :.2f}: egim <=0, veri ile ULASILAMAZ")
    with io .open ("results/p1_ogrenme_egrisi.json","w",encoding ="utf-8")as fjs :
        json .dump ({"egri":{str (k ):v for k ,v in EGRI .items ()},
        "log_havuz":list (map (float ,kh )),"log_uretici":list (map (float ,ku )),
        "egim_son":float (egim_son ),"simdiki_grup":simdi_g ,
        "not":("Alt-ornekleme GRUP bazinda (ikizler birlikte). Ekstrapolasyon "
        "yalniz egri doymuyorsa anlamlidir; son iki noktanin egimi "
        "raporlanir.")},fjs ,indent =1 )
    print ("receipt -> results/p1_ogrenme_egrisi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
