# -*- coding: utf-8 -*-
"""REJIM ESIGININ KARARLILIGI: 90 a bicak sirti mi?

SORUN. Rejim kapisi `n01 >= 90 -> P6`. D7'de parcalarin n01 ortalamasi 89.6 --
i.e. threshold full dagilimin ortasinda and small a offset yonlendirmeyi tersine
cevirebiliyor. Nitekim CWT (n01=86) tabana, WIE (n01=141) P6'ya gidiyor and
IKISI DE wrong tarafta.

Esigi D7'ye bakarak oynatmak SINAVDAN AYAR CEKMEKTIR. Bunun instead of here
`full` korpusunun MARKA KATLARINDA threshold EGRISI cikarilir:

  * each threshold degeri for fold-disi robot F1
  * egrinin DUZ oldugu a bant present mi (kararlilik)
  * most iyi threshold with 90'in farki anlamli mi

Egri duzse threshold onemsizdir and bicak sirti korkusu yersizdir. Egri sivriyse
threshold kirilgandir and YUMUSAK GECIS (two kolun birlesimi ya da bant inside
tabana yaslanma) is required.
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
import canonical_d7 as K # noqa: E402
import product_genis # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from run_p6_rejim import p6_cikti ,taban_cikti # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

PAKET =os .environ .get ("P6_MODEL","results/p6_kademe2_model.pkl")
ESIKLER =list (range (40 ,200 ,10 ))


def say (P ,D ,d ):
    return match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,K .ACI ,
    False ,signed =True )[:3 ]


def puanla (data_ ,sel_ ):
    T =[0 ,0 ,0 ]
    for d ,p6 in zip (data_ ,sel_ ):
        c =d ["_p6_c"]if p6 else d ["_tb_c"]
        for i in range (3 ):
            T [i ]+=c [i ]
    return 2 *T [0 ]/max (2 *T [0 ]+T [1 ]+T [2 ],1 )


def main ():
    t0 =time .time ()
    pk =pickle .load (open (PAKET ,"rb"))
    tb =product_genis .model_yukle ()
    data_ =[]
    for cluster in os .environ .get ("P6_KUME","tam,d6").split (","):
        data_ +=yukle (cluster .strip (),int (os .environ .get ("P6_TR","0")))
    print (f"{len (data_ )} part ({time .time ()-t0 :.0f} s)",flush =True )
    for i ,d in enumerate (data_ ,1 ):
        (Pt ,Dt ),_s =taban_cikti (d ,tb )
        Pp ,Dp =p6_cikti (d ,pk )
        d ["_tb_c"]=say (Pt ,Dt ,d )
        d ["_p6_c"]=say (Pp ,Dp ,d )
        d ["_n01"]=float ((d ["kaynak"]!=2 ).sum ())
        if i %600 ==0 :
            print (f"  {i }/{len (data_ )} ({time .time ()-t0 :.0f} s)",flush =True )

    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=200 ]
    print (f"katlar: {katlar }\n",flush =True )
    print (f"{'threshold':>6}{'fold-disi robot':>16}{'P6 orani':>10}")
    egri ={}
    for e in ESIKLER :
        T =[0 ,0 ,0 ]
        p6n =tot =0 
        for b in katlar :
            dis =[d for d in data_ if d ["mfg"]==b ]
            for d in dis :
                c =d ["_p6_c"]if d ["_n01"]>=e else d ["_tb_c"]
                for i in range (3 ):
                    T [i ]+=c [i ]
                p6n +=int (d ["_n01"]>=e )
                tot +=1 
        f1 =2 *T [0 ]/max (2 *T [0 ]+T [1 ]+T [2 ],1 )
        egri [e ]={"robot":f1 ,"p6_orani":p6n /max (tot ,1 )}
        print (f"{e :>6}{f1 :>16.4f}{p6n /max (tot ,1 ):>10.2f}")

    en =max (egri ,key =lambda k :egri [k ]["robot"])
    vtx_ =egri [en ]["robot"]
    bant =[e for e in ESIKLER if egri [e ]["robot"]>=vtx_ -0.01 ]
    print (f"\nEN IYI threshold {en } -> {vtx_ :.4f}")
    print (f"TEPEDEN 0.01 ICINDE kalan esikler: {bant }")
    print (f"90'in degeri: {egri .get (90 ,{}).get ('robot',0 ):.4f} "
    f"(tepeden {vtx_ -egri .get (90 ,{}).get ('robot',0 ):+.4f})")
    print ("YORUM: bant genisse threshold KARARLI, dar whereas KIRILGAN.")
    json .dump ({"damga":makbuz_hash .damga (),"egri":egri ,"en_iyi":en ,
    "bant":bant ,"katlar":katlar ,"n_parca":len (data_ ),
    "not":"Rejim esigi kararlilik egrisi. Kat-disi measurement; D7'ye "
    "BAKILMADI. Esik secimi bu egriden yapilir."},
    open ("results/rejim_kararlilik.json","w"),indent =1 )
    print (f"receipt -> results/rejim_kararlilik.json ({time .time ()-t0 :.0f} s)")


if __name__ =="__main__":
    main ()
