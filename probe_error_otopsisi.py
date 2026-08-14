# -*- coding: utf-8 -*-
"""HATA OTOPSISI — loss NEREYE gidiyor, cevrimdisi

WHY. Sunum sayilarinda two large loss present and ikisinin de SEBEBI
olculmedi:
    tespit 0.7878 -> robot axis 0.5764   (-0.2114)
    precision 0.5008                        (ciktinin YARISI wrong)

Onarilacak seyi bilmeden onarim denemek tahmindir. Bu betik dokumden
each FP and FN'i SINIFLANDIRIR:

FP (wrong pozitif) turleri:
  double      : a TP'ye 5mm'den yakin -- AYNI acikligin ikinci kopyasi
  angle       : a GT'ye lateral as yakin but ACI tutmuyor
  sign    : GT with same EKSENDE but TERS yonde (180 derece)
  hayalet   : no GT'ye yakin not -- real wrong tespit

FN (kacan) turleri:
  yakin_var : next to prediction VAR but kabul kutusuna girmiyor (angle/lateral)
  empty       : yakininda no prediction YOK -- pool/tespit sorunu

Her turun PAYI, hangi onarimin ne up to getirecegini soyler.
D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

DOKUM =os .environ .get ("HO_DOKUM","results/_tahmin_dokumu.json")
YOL =os .environ .get ("HO_YOL","saha")
CIFT_R =float (os .environ .get ("HO_CIFT","5.0"))


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def main ():
    d =[r for r in json .load (open (DOKUM ))if r ["yol"]==YOL ]
    if not d :
        sys .exit (f"{DOKUM } icinde '{YOL }' yok")
    fp_tur =collections .Counter ()
    fn_tur =collections .Counter ()
    tp_top =fp_top =fn_top =0 
    for r in d :
        P =np .asarray (r ["P"],float ).reshape (-1 ,3 )
        D =_birim (np .asarray (r ["D"],float ).reshape (-1 ,3 ))
        G =np .asarray (r ["G"],float )
        Gd =_birim (np .asarray (r ["Gd"],float ))
        dg =float (r ["diag"])
        if not len (G ):
            continue 
        sonuc =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
        signed =True )
        tp ,fp ,fn =sonuc [:3 ]
        tp_top +=tp ;fp_top +=fp ;fn_top +=fn 
        if not len (P ):
            fn_tur ["bos"]+=len (G )
            continue 
            # hangi prediction/GT eslesti: eslesme ayrintisi otherwise yeniden kur
            # (kaba: each GT for kabul kutusuna giren EN YAKIN prediction)
        v =P [:,None ,:]-G [None ,:,:]
        al =np .einsum ("pgc,gc->pg",v ,Gd )
        yan =np .linalg .norm (v -al [...,None ]*Gd [None ,:,:],axis =-1 )
        aci =np .degrees (np .arccos (np .clip (D @Gd .T ,-1 ,1 )))
        kabul =(yan <=K .YANAL )&(np .abs (al )<=40.0 )&(aci <=K .ACI )
        p_esli =kabul .any (1 )
        g_esli =kabul .any (0 )
        # --- FN siniflandirmasi
        for j in np .where (~g_esli )[0 ]:
            yakin =(yan [:,j ]<=K .YANAL )&(np .abs (al [:,j ])<=40.0 )
            fn_tur ["yakin_var"if yakin .any ()else "bos"]+=1 
            # --- FP siniflandirmasi
        tp_nok =P [p_esli ]
        for i in np .where (~p_esli )[0 ]:
            if len (tp_nok )and np .min (
            np .linalg .norm (tp_nok -P [i ],axis =1 ))<=CIFT_R :
                fp_tur ["cift"]+=1 
                continue 
            yakin =(yan [i ]<=K .YANAL )&(np .abs (al [i ])<=40.0 )
            if yakin .any ():
                j =np .where (yakin )[0 ]
                a_min =aci [i ,j ].min ()
                ters =np .degrees (np .arccos (np .clip (
                np .abs (D [i ]@Gd [j ].T ),-1 ,1 ))).min ()
                if ters <=K .ACI and a_min >K .ACI :
                    fp_tur ["sign"]+=1 
                else :
                    fp_tur ["aci"]+=1 
            else :
                fp_tur ["hayalet"]+=1 

    print (f"{len (d )} part | yol={YOL }")
    print (f"TP {tp_top } | FP {fp_top } | FN {fn_top }")
    f1 =2 *tp_top /max (2 *tp_top +fp_top +fn_top ,1 )
    print (f"robot ISARETLI F1 {f1 :.4f}\n")
    tf =max (sum (fp_tur .values ()),1 )
    print (f"--- YANLIS POZITIF dagilimi (toplam {sum (fp_tur .values ())}) ---")
    for k ,v in fp_tur .most_common ():
        print (f"  {k :<10}{v :>6}  {v /tf :>7.1%}")
    tn =max (sum (fn_tur .values ()),1 )
    print (f"\n--- KACAN dagilimi (toplam {sum (fn_tur .values ())}) ---")
    for k ,v in fn_tur .most_common ():
        print (f"  {k :<10}{v :>6}  {v /tn :>7.1%}")
    print ("\nOKUMA:")
    print ("  cift YUKSEK      -> birlestirme/NMS yaricapi ONARILABILIR")
    print ("  sign YUKSEK    -> sign kurali ONARILABILIR")
    print ("  aci YUKSEK       -> direction tahmini sorunu (model isi)")
    print ("  hayalet YUKSEK   -> pool/gate sorunu")
    print ("  FN yakin_var     -> tahmin VAR, kabul kutusuna sokulamiyor")
    print ("  FN bos           -> pool o acikligi HIC uretmemis")
    json .dump ({"yol":YOL ,"n_parca":len (d ),"tp":tp_top ,"fp":fp_top ,
    "fn":fn_top ,"f1":f1 ,"fp_tur":dict (fp_tur ),
    "fn_tur":dict (fn_tur ),
    "not":"Hata otopsisi, cevrimdisi. D7'ye BAKILMADI."},
    open (f"results/hata_otopsisi_{YOL }.json","w"),indent =1 )
    print (f"receipt -> results/hata_otopsisi_{YOL }.json")


if __name__ =="__main__":
    main ()
