# -*- coding: utf-8 -*-
"""REJIM YONLENDIRICI v2: TEK ESIK instead of OGRENILMIS karar.

WHY. v1 single a istatistige single threshold koyuyordu (`n01 >= 90 -> P6`). D7'de
sonuc: 8/12 markada arti but WIE **-0.0802**, A-B/CCD/KLM ~-0.03. Yani gate
some parcalarda wrong tarafa yonlendiriyor and single threshold bunu ayirt edemiyor.

Bu betik part duzeyinde KUCUK a siniflandirici ogrenir:
    input   : inference aninda gorulebilen part istatistikleri
    label  : P6 mi TABAN mi DAHA IYI (part basina F1 farki)
    output   : P6 kullan / TABAN kullan

Etiket GT'den turer -- but this EGITIM verisidir, exam not. Urun only
istatistikleri gorur. Kural yine `full` MARKA KATLARINDA dogrulanir: disarida
birakilan markada router TARANMAZ.

DURUSTLUK: v1'in D7 count (0.3115) ILAN EDILMIS okumaydi. Bu v2, D7'ye
BAKILMADAN `full` katlarinda olculur; dagitilirsa D7 okuma #2 harcanir.
"""
import collections 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_u25")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
import product_genis # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from run_p6_rejim import p6_cikti ,taban_cikti # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

PAKET =os .environ .get ("P6_MODEL","results/p6_kademe2_model.pkl")
OZ_AD =["n01","nmesh","mesh_oran","n_aday","n_secenek",
"taban_maks","taban_ort3","taban_n_gecen",
"kutu_en","kutu_boy","kutu_yuk","kutu_hacim","tepe_sayi"]


def oznitelik (d ,s_tb ,V ):
    k =d ["kaynak"]
    n01 =int ((k !=2 ).sum ())
    nm =int ((k ==2 ).sum ())
    s =np .sort (np .asarray (s_tb ,float ))[::-1 ]if len (s_tb )else np .zeros (1 )
    b =np .sort (V .max (0 )-V .min (0 ))if len (V )else np .zeros (3 )
    return [float (n01 ),float (nm ),nm /max (n01 ,1 ),float (len (d ["P"])),
    float (len (d ["idx"])),float (s [0 ]),float (s [:3 ].mean ()),
    float ((np .asarray (s_tb ,float )>=product_genis .ESIK ).sum ()),
    float (b [0 ]),float (b [1 ]),float (b [2 ]),
    float (b [0 ]*b [1 ]*b [2 ]),float (len (V ))]


def f1p (P ,D ,d ):
    tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,K .ACI ,
    False ,signed =True )[:3 ]
    return 2 *tp /max (2 *tp +fp +fn ,1 ),(tp ,fp ,fn )


def puanla (veri ,secim ):
    per =collections .defaultdict (lambda :[0 ,0 ,0 ])
    for d ,p6 in zip (veri ,secim ):
        t =d ["_p6_c"]if p6 else d ["_tb_c"]
        q =per [d ["mfg"]]
        for i in range (3 ):
            q [i ]+=t [i ]
    pm ={m :2 *q [0 ]/max (2 *q [0 ]+q [1 ]+q [2 ],1 )for m ,q in per .items ()}
    T =[sum (q [i ]for q in per .values ())for i in range (3 )]
    return {"robot":2 *T [0 ]/max (2 *T [0 ]+T [1 ]+T [2 ],1 ),
    "makro":float (np .mean (list (pm .values ())))if pm else 0.0 ,
    "en_kotu":float (min (pm .values ()))if pm else 0.0 ,
    "TP":T [0 ],"FP":T [1 ],"FN":T [2 ]}


def main ():
    t0 =time .time ()
    pk =pickle .load (open (PAKET ,"rb"))
    tb_model =product_genis .model_yukle ()
    mesh_diz ={"tam":"results/_p1_olasilik_brepegit",
    "d6":"results/_p1_olasilik"}
    veri =[]
    for cluster in os .environ .get ("P6_KUME","tam,d6").split (","):
        cluster =cluster .strip ()
        for d in yukle (cluster ,int (os .environ .get ("P6_TR","0"))):
            d ["_kume"]=cluster 
            veri .append (d )
    print (f"{len (veri )} part ({time .time ()-t0 :.0f} s)",flush =True )

    X ,Y ,W =[],[],[]
    for i ,d in enumerate (veri ,1 ):
        (Pt ,Dt ),s_tb =taban_cikti (d ,tb_model )
        Pp ,Dp =p6_cikti (d ,pk )
        f_t ,c_t =f1p (Pt ,Dt ,d )
        f_p ,c_p =f1p (Pp ,Dp ,d )
        d ["_tb_c"],d ["_p6_c"]=c_t ,c_p 
        mf =f"{mesh_diz [d ['_kume']]}/{d ['pid']}.npz"
        V =(np .asarray (np .load (mf )["V"],float )
        if os .path .exists (mf )else np .zeros ((0 ,3 )))
        X .append (oznitelik (d ,s_tb ,V ))
        Y .append (int (f_p >f_t ))
        W .append (abs (f_p -f_t ))# farkin BUYUKLUGU up to onemli
        if i %500 ==0 :
            print (f"  {i }/{len (veri )} ({time .time ()-t0 :.0f} s)",flush =True )
    X =np .asarray (X ,float )
    Y =np .asarray (Y ,int )
    W =np .asarray (W ,float )
    print (f"P6 daha iyi olan part: {Y .mean ():.3f} | ortalama |fark| "
    f"{W .mean ():.4f}",flush =True )

    brand =collections .Counter (d ["mfg"]for d in veri )
    katlar =[m for m ,n in brand .items ()if n >=200 ]
    print (f"katlar: {katlar }",flush =True )

    hep_t =puanla (veri ,[False ]*len (veri ))
    hep_p =puanla (veri ,[True ]*len (veri ))
    v1 =puanla (veri ,[X [i ,0 ]>=90.0 for i in range (len (veri ))])
    print (f"\nHEPSI TABAN  robot {hep_t ['robot']:.4f} makro {hep_t ['makro']:.4f}")
    print (f"HEPSI P6     robot {hep_p ['robot']:.4f} makro {hep_p ['makro']:.4f}")
    print (f"v1 (n01>=90) robot {v1 ['robot']:.4f} makro {v1 ['makro']:.4f}")

    top =collections .Counter ()
    for b in katlar :
        ic =[i for i ,d in enumerate (veri )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (veri )if d ["mfg"]==b ]
        m =HistGradientBoostingClassifier (
        max_iter =200 ,learning_rate =0.06 ,max_leaf_nodes =15 ,
        l2_regularization =1.0 ,random_state =0 ).fit (
        X [ic ],Y [ic ],sample_weight =np .maximum (W [ic ],1e-4 ))
        p =m .predict_proba (X [dis ])[:,1 ]>=0.5 
        r =puanla ([veri [i ]for i in dis ],list (p ))
        for k_ in ("TP","FP","FN"):
            top [k_ ]+=r [k_ ]
        v1b =puanla ([veri [i ]for i in dis ],[X [i ,0 ]>=90.0 for i in dis ])
        print (f"  {b :<6} v1 {v1b ['robot']:.4f} -> v2 {r ['robot']:.4f} "
        f"({r ['robot']-v1b ['robot']:+.4f}) | P6 secim orani "
        f"{p .mean ():.2f}",flush =True )
    f1 =2 *top ["TP"]/max (2 *top ["TP"]+top ["FP"]+top ["FN"],1 )
    v1k =puanla ([veri [i ]for i ,d in enumerate (veri )if d ["mfg"]in katlar ],
    [X [i ,0 ]>=90.0 for i ,d in enumerate (veri )
    if d ["mfg"]in katlar ])
    print (f"\nKAT-DISI HAVUZLANMIS: v1 {v1k ['robot']:.4f} -> v2 {f1 :.4f} "
    f"({f1 -v1k ['robot']:+.4f})")

    son =HistGradientBoostingClassifier (
    max_iter =200 ,learning_rate =0.06 ,max_leaf_nodes =15 ,
    l2_regularization =1.0 ,random_state =0 ).fit (
    X ,Y ,sample_weight =np .maximum (W ,1e-4 ))
    pk ["rejim2"]={"model":son ,"oz_ad":OZ_AD ,"threshold":0.5 }
    with open (PAKET .replace (".pkl","_rejim2.pkl"),"wb")as f :
        pickle .dump (pk ,f )
    json .dump ({"damga":makbuz_hash .damga (),
    "hepsi_taban":hep_t ,"hepsi_p6":hep_p ,"v1_tek_esik":v1 ,
    "v2_kat_disi_robot":f1 ,"v1_kat_disi_robot":v1k ["robot"],
    "katlar":katlar ,"n_parca":len (veri ),
    "p6_daha_iyi_orani":float (Y .mean ()),
    "not":"Ogrenilmis regime router. Etiket EGITIM verisinden "
    "(part basina F1 farki); urun yalniz istatistikleri "
    "gorur. D7'ye BAKILMADI. HEPSI/v1 satirlari ORNEKLEM-ICI."},
    open ("results/rejim2.json","w"),indent =1 )
    print (f"receipt -> results/rejim2.json  ({time .time ()-t0 :.0f} s)")


if __name__ =="__main__":
    main ()
