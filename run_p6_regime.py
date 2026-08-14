# -*- coding: utf-8 -*-
"""REJIM YONLENDIRME: part basina "P6 mu, DAGITILAN TABAN mi?"

FINDING (real training, brand katlari):
    WEI  n=686  TABAN 0.2797 -> P6 0.3795   (+0.0998)
    PXC  n=713  TABAN 0.5850 -> P6 0.3972   (-0.1878)
PXC parcalarinda n01 (mesh DISI candidate) only 43; uzerine 96 mesh eklenince
pool uce katlaniyor and ZATEN IYI CALISAN tabani boguyor. WEI'de n01=121 and
mesh orani very more ilimli, orada P6 kazaniyor.

Yani mesh havuzu HER PARCADA correct arac not. Bu betik, inference aninda
GORULEBILEN a istatistige according to yonlendirme kurali ogrenir:

    istatistik(part) >= threshold  ->  P6 kullan
    aksi halde                 ->  DAGITILAN TABAN (product_wide) kullan

URUNDEKI KARSILIGI TEK SATIR: `product_p6.output` esigi gecmeyen parcada `None`
returns; `canonical_chain` already old yola duser. Yani yonlendirme new a kod
yolu ACMAZ, present which is geri-dusmeyi KURALLI hale getirir.

Aday istatistikler (all of them etiketsiz, inference aninda bilinir):
    n01        mesh disi candidate count
    mesh_oran  mesh / n01
    n_aday     total candidate
    n_secenek  total (konum x direction) secenegi

Esik `full` korpusunun MARKA KATLARINDA secilir (disarida birakilan markada
TARANMAZ). Olcut MAKRO -- single markada cokmeyen kurali tercih eder.
"""
import collections 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_u25")
sys .path .insert (0 ,".")
import lattice # noqa: E402
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
import product_wide # noqa: E402
import wire_gate # noqa: E402
import direction_bank as YB # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

AB =p6_decision .AB 
C0 =AB 
PAKET =os .environ .get ("P6_MODEL","results/p6_kademe2_model.pkl")
# YONLENDIRME ISTATISTIKLERI -- all of them inference aninda, ETIKETSIZ bilinir.
# Ilk dordu pool buyuklugu (zayif vekil). Son ucu TABANIN KENDI GUVENI:
# desen "P6, tabanin ZAYIF oldugu places kazanir" oldugu for most dogrudan
# yonlendirme sinyali tabanin skorlaridir -- part on high safe
# tespit uretiyorsa ona dokunma, uretmiyorsa P6'ya gec.
ISTATISTIKLER =("n01","mesh_oran","n_aday","n_secenek",
"taban_maks_ters","taban_ort3_ters","taban_sayi_ters")


def istatistik (d ,s_tb ):
    k =d ["source"]
    n01 =int ((k !=2 ).sum ())
    nm =int ((k ==2 ).sum ())
    s =np .sort (np .asarray (s_tb ,float ))[::-1 ]if len (s_tb )else np .zeros (1 )
    top3 =float (s [:3 ].mean ())
    # "_ters": rule HER ZAMAN `value >= threshold -> P6` seklinde; tabanin guveni
    # DUSUKKEN P6 istedigimiz for isareti ters ceviriyoruz.
    return {"n01":float (n01 ),"mesh_oran":nm /max (n01 ,1 ),
    "n_aday":float (len (d ["P"])),"n_secenek":float (len (d ["idx"])),
    "taban_maks_ters":-float (s [0 ]),
    "taban_ort3_ters":-top3 ,
    "taban_sayi_ters":-float ((np .asarray (s_tb ,float )>=
    product_wide .ESIK ).sum ())}


def taban_cikti (d ,model ):
    """DAGITILAN yolun ciktisi, onbellekten yeniden kurulmus.

    `product_wide.sec` with same: A+B, part-ici z-skor, HGB-derin, threshold 0.05,
    kalabalik NMS, sign correction. Mesh adaylari tabanin havuzunda YOK.
    """
    k =np .where (d ["X"][:,C0 ]==1.0 )[0 ]
    k =k [d ["source"][d ["idx"][k ]]!=2 ]
    empty_ =(np .zeros ((0 ,3 )),np .zeros ((0 ,3 )))
    if not len (k ):
        return empty_ ,np .zeros (0 )
    ci =d ["idx"][k ]
    X =d ["X"][k ][:,:AB ]
    s =np .asarray (model .predict_proba (
    wire_gate .within_part (X ,"zskor"))[:,1 ],float )
    m =s >=product_wide .ESIK 
    if not m .any ():
        return empty_ ,s 
    P ,D =d ["P"][ci [m ]],d ["D"][ci [m ]]
    T =d ["X"][k ][m ][:,AB +len (YB .OZ_AD ):]
    if len (P )>1 :
        nm =wire_gate .crowd_mask (P ,s [m ])
        P ,D ,T =P [nm ],D [nm ],T [nm ]
    return (P ,product_wide .isaret_duzelt (D ,T )),s 


def p6_cikti (d ,pk ):
    Xd =np .hstack ([p6_decision .donustur (d ["X"],pk .get ("zskor","ab")),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])])
    if pk .get ("arm")=="P6_GEO":
        Xd =np .hstack ([Xd [:,58 :AB ],Xd [:,AB :]])
    s =np .asarray (pk ["kademe1"].predict_proba (
    Xd .astype (np .float32 ))[:,1 ],float )
    if pk .get ("kademe2")is not None :
        Pt ,Dt =p6_decision .sec_ayrintili (
        d ["P"],d ["idx"],d ["YD"],s ,tuple (pk ["tohum_kural"]),
        nms_mm =float (pk ["tohum_nms"]))[:2 ]
        kb =lattice .oznitelik (d ["P"][d ["idx"]],d ["YD"],Pt ,Dt )
        kk =np .where (s >=float (pk .get ("kisa_esik",0.20 )))[0 ]
        s2 =np .zeros (len (s ))
        if len (kk ):
            X2 =np .hstack ([Xd [kk ],kb [kk ],s [kk ][:,None ]])
            s2 [kk ]=pk ["kademe2"].predict_proba (X2 .astype (np .float32 ))[:,1 ]
        s =s2 
    return p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,tuple (pk ["rule"]),
    nms_mm =float (pk ["nms"]))


def puanla (data_ ,sel_ ):
    """`secim[i]` True whereas P6 ciktisi, False whereas TABAN ciktisi is used."""
    per =collections .defaultdict (lambda :[0 ,0 ,0 ])
    for d ,p6 in zip (data_ ,sel_ ):
        P ,D =d ["_p6"]if p6 else d ["_tb"]
        a ,b ,c =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,K .ACI ,
        False ,signed =True )[:3 ]
        q =per [d ["mfg"]]
        q [0 ]+=a ;q [1 ]+=b ;q [2 ]+=c 
    pm ={m :2 *q [0 ]/max (2 *q [0 ]+q [1 ]+q [2 ],1 )for m ,q in per .items ()}
    T =[sum (q [i ]for q in per .values ())for i in range (3 )]
    return {"robot":2 *T [0 ]/max (2 *T [0 ]+T [1 ]+T [2 ],1 ),
    "makro":float (np .mean (list (pm .values ())))if pm else 0.0 ,
    "TP":T [0 ],"FP":T [1 ],"FN":T [2 ],"brand":pm }


def main ():
    t0 =time .time ()
    pk =pickle .load (open (PAKET ,"rb"))
    print (f"paket: arm {pk .get ('arm')} | kural {pk .get ('rule')} | "
    f"nms {pk .get ('nms')} | 2.kademe "
    f"{'VAR'if pk .get ('kademe2')is not None else 'YOK'}",flush =True )
    tb_model =product_wide .model_yukle ()
    if tb_model is None :
        sys .exit ("dagitilan gate modeli none")

    data_ =[]
    for cluster in os .environ .get ("P6_KUME","tam,d6").split (","):
        data_ +=yukle (cluster .strip (),int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
    print (f"{len (data_ )} part yuklendi ({time .time ()-t0 :.0f} s)",flush =True )

    for i ,d in enumerate (data_ ,1 ):
        d ["_tb"],s_tb =taban_cikti (d ,tb_model )
        d ["_p6"]=p6_cikti (d ,pk )
        d ["_ist"]=istatistik (d ,s_tb )
        if i %400 ==0 :
            print (f"  cikti {i }/{len (data_ )} ({time .time ()-t0 :.0f} s)",
            flush =True )

    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=200 ]
    print (f"brand katlari: {katlar }",flush =True )

    hep_tb =puanla (data_ ,[False ]*len (data_ ))
    hep_p6 =puanla (data_ ,[True ]*len (data_ ))
    print (f"\nHEPSI TABAN : robot {hep_tb ['robot']:.4f} makro {hep_tb ['makro']:.4f}")
    print (f"HEPSI P6    : robot {hep_p6 ['robot']:.4f} makro {hep_p6 ['makro']:.4f}")

    # --- fold inside threshold secimi, disarida birakilan markada measurement ----------
    res_ ={}
    for ist in ISTATISTIKLER :
        v =np .array ([d ["_ist"][ist ]for d in data_ ])
        candidates =[-np .inf ]+list (np .percentile (v ,[10 ,20 ,30 ,40 ,50 ,60 ,
        70 ,80 ,90 ]))+[np .inf ]
        top =collections .Counter ()
        secilen =[]
        for b in katlar :
            ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
            dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
            IC =[data_ [i ]for i in ic ]
            en =max (candidates ,key =lambda e :puanla (
            IC ,[data_ [i ]["_ist"][ist ]>=e for i in ic ])["makro"])
            r =puanla ([data_ [i ]for i in dis ],
            [data_ [i ]["_ist"][ist ]>=en for i in dis ])
            for k_ in ("TP","FP","FN"):
                top [k_ ]+=r [k_ ]
            secilen .append (en )
        f1 =2 *top ["TP"]/max (2 *top ["TP"]+top ["FP"]+top ["FN"],1 )
        res_ [ist ]={"robot":f1 ,"esikler":[float (x )for x in secilen ],
        **dict (top )}
        print (f"{ist :<11} fold-disi robot {f1 :.4f} | secilen esikler "
        f"{[round (float (x ),2 )for x in secilen ]}",flush =True )

    en_ist =max (res_ ,key =lambda k :res_ [k ]["robot"])
    # threshold: katlarda secilenlerin MEDYANI (single kata asiri uymasin)
    en_esik =float (np .median (res_ [en_ist ]["esikler"]))
    print (f"\nSECILEN: {en_ist } >= {en_esik :.3f} -> P6, altinda TABAN")
    print (f"  fold-disi robot {res_ [en_ist ]['robot']:.4f}  "
    f"(hepsi-baseline {hep_tb ['robot']:.4f}, hepsi-P6 {hep_p6 ['robot']:.4f})")

    pk ["regime"]={"istatistik":en_ist ,"threshold":en_esik }
    with open (PAKET ,"wb")as f :
        pickle .dump (pk ,f )
    json .dump ({"damga":makbuz_hash .damga (),"hepsi_taban":hep_tb ,
    "hepsi_p6":hep_p6 ,"istatistikler":res_ ,
    "secilen":{"istatistik":en_ist ,"threshold":en_esik },
    "katlar":katlar ,"n_parca":len (data_ ),
    "not":"Rejim yonlendirme: part basina P6 mu TABAN mi. Esik "
    "brand katlarinda MAKRO olcutle secildi; disarida "
    "birakilan markada TARANMADI. D7'ye BAKILMADI."},
    open ("results/p6_rejim.json","w"),indent =1 )
    print (f"receipt -> results/p6_rejim.json  ({time .time ()-t0 :.0f} s)")


if __name__ =="__main__":
    main ()
