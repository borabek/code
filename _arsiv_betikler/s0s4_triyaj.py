# -*- coding: utf-8 -*-
"""S0+S4: DEFECT HARITASI and TRIYAJ -- onarimi HERKESE not, KIME uygulamali?

S3 SONUCU: onarim HERKESE uygulanınca fiziksel kusur %18.7 -> %10.2 (iyi) AMA metrik
kayitsiz: tespit +0.0016, robot -0.0053; gecisler TP->FP 12 / FP->TP 9. Onceden yazilan
kill GECMEDI.

REASON MEASURED: manufacturer CP'lerinin ~%47'si GOVDENIN ICINDE (kontakta). Noktayi agza
tasimak fiziksel gecerliligi kazandirirken GT'den UZAKLASTIRIYOR. "Robot for correct"
with "ureticiye yakin" this noktada CELISIYOR.

TRIYAJ FIKRI: onarim herkese not, DEFECT YOGUNLUGU YUKSEK olanlara uygulansin.
A4 olctu: bayrak count arttikca FP olma olasiligi artiyor (zenginlesme 2-3x). Uc bayragi
birden yanan 36 CP large olasilikla cop; single bayrakli olanlar onarilabilir TP may be.

Bu betik onbellekten (S2 ciktisi) works, YENI mesh islemi YOK. Uc triyaj denenir:
    T1  only >=1 bayrakli (= S3'un yaptigi, baseline)
    T2  only >=2 bayrakli
    T3  only 3 bayrakli
    T4  >=2 bayrakli ONAR + 3 bayrakli RED
KILL (S4, onceden yazildi): tespit >= +0.005 VE fiziksel kusur orani >= -%5 mutlak.
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import measure_set 
from sina_cluster import match_greedy ,f1w 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    DER ,gate ,ek =T2 .yukle ()
    with open ("results/_s2_onarilmis.pkl","rb")as f :
        R =pickle .load (f )

        # ---------- S0: DEFECT HARITASI
    say ={}
    tpfp ={}
    for r in DER :
        d =R .get (r ["pid"])
        if d is None :
            continue 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        _ ,_ ,_ ,b =match_greedy (d ["P0"],d ["D0"],G ,Gd ,r ["diag"],0.0 ,180.0 ,True )
        es ={e [0 ]for e in b ["eslesme"]}
        for i ,bb in enumerate (d ["B0"]):
            k =sum (bb )
            say [k ]=say .get (k ,0 )+1 
            t =tpfp .setdefault (k ,[0 ,0 ])
            t [0 if i in es else 1 ]+=1 
    print (f"{'='*72 }\nS0 -- KUSUR YOGUNLUGU HARITASI\n{'='*72 }")
    print (f"{'bayrak count':<16}{'CP':>7}{'TP':>7}{'FP':>7}{'FP orani':>11}")
    for k in sorted (say ):
        t ,f =tpfp [k ]
        print (f"{k :<16}{say [k ]:>7}{t :>7}{f :>7}{100 *f /max (t +f ,1 ):>10.1f}%")
    print ("  -> bayrak count arttikca FP olma olasiligi artiyorsa triyaj MESRU")

    # ---------- S4: TRIYAJ
    def kos (en_az ,red_esigi =None ):
        """en_az bayrakli olanlar ONARILIR; red_esigi up to bayrakli which is ELENIR."""
        rows_d ,rows_r =[],[]
        t2f =f2t =0 
        kus =ntop =0 
        for r in DER :
            d =R .get (r ["pid"])
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="very"if r ["n"]>=8 else "low"
            if d is None :
                P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
            else :
                P =d ["P0"].copy ();D =d ["D0"].copy ()
                tut =np .ones (len (P ),bool )
                for i ,bb in enumerate (d ["B0"]):
                    k =sum (bb )
                    if red_esigi is not None and k >=red_esigi :
                        tut [i ]=False 
                        continue 
                    if k >=en_az :
                        P [i ]=d ["P1"][i ];D [i ]=d ["D1"][i ]
                        # onarim/red SONRASI kusur sayimi
                for i in range (len (P )):
                    if not tut [i ]:
                        continue 
                    ntop +=1 
                    bb =d ["B1"][i ]if sum (d ["B0"][i ])>=en_az else d ["B0"][i ]
                    kus +=any (bb )
                P =P [tut ];D =D [tut ]
            _ ,_ ,_ ,b0 =match_greedy (d ["P0"]if d else P ,d ["D0"]if d else D ,G ,Gd ,
            r ["diag"],0.0 ,180.0 ,True )
            tp ,fp ,fn ,b1 =match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )
            rows_d .append ((rj ,tp ,fp ,fn ))
            tpr ,fpr ,fnr ,_ =match_greedy (P ,D ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,
            signed =True )
            rows_r .append ((rj ,tpr ,fpr ,fnr ))
            e0 ={e [0 ]for e in b0 ["eslesme"]};e1 ={e [0 ]for e in b1 ["eslesme"]}
            t2f +=len (e0 -e1 );f2t +=len (e1 -e0 )
        return (f1w (rows_d ),f1w (rows_r ),t2f ,f2t ,
        kus /max (ntop ,1 ),ntop )

    print (f"\n{'='*72 }\nS4 -- TRIYAJ (onbellekten, yeni mesh islemi yok)\n{'='*72 }")
    # baseline: never onarim absent
    tb_d ,tb_r ,_ ,_ ,tb_k ,_ =kos (en_az =99 )
    print (f"{'triyaj':<28}{'tespit':>9}{'robot':>9}{'kusur%':>9}{'TP->FP':>8}{'FP->TP':>8}")
    print (f"{'TABAN (onarim absent)':<28}{tb_d :>9.4f}{tb_r :>9.4f}{100 *tb_k :>8.1f}%{'-':>8}{'-':>8}")
    SON ={}
    for ad ,kw in (("T1 >=1 bayrak ONAR",dict (en_az =1 )),
    ("T2 >=2 bayrak ONAR",dict (en_az =2 )),
    ("T3  =3 bayrak ONAR",dict (en_az =3 )),
    ("T4 >=2 ONAR, =3 RED",dict (en_az =2 ,red_esigi =3 ))):
        d_ ,r_ ,t2f ,f2t ,kk ,ntop =kos (**kw )
        SON [ad ]={"tespit":d_ ,"robot":r_ ,"kusur":kk ,"tp2fp":t2f ,"fp2tp":f2t }
        print (f"{ad :<28}{d_ :>9.4f}{r_ :>9.4f}{100 *kk :>8.1f}%{t2f :>8}{f2t :>8}")

    print (f"\nKILL (S4): tespit >= baseline+0.005 VE kusur orani <= baseline-%5 mutlak")
    for ad ,v in SON .items ():
        g1 =v ["tespit"]>=tb_d +0.005 
        g2 =v ["kusur"]<=tb_k -0.05 
        print (f"  {ad :<28} tespit {v ['tespit']-tb_d :+.4f} ({'OK'if g1 else 'x'})  "
        f"kusur {100 *(v ['kusur']-tb_k ):+.1f}pp ({'OK'if g2 else 'x'})  "
        f"-> {'GECTI'if (g1 and g2 )else 'GECMEDI'}")

    with io .open ("results/s0s4_triyaj.json","w",encoding ="utf-8")as f :
        json .dump ({"harita":{str (k ):{"cp":say [k ],"tp":tpfp [k ][0 ],"fp":tpfp [k ][1 ]}
        for k in sorted (say )},
        "baseline":{"tespit":tb_d ,"robot":tb_r ,"kusur":tb_k },
        "triyaj":SON },f ,indent =1 ,ensure_ascii =False )
    print ("\nmakbuz -> results/s0s4_triyaj.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
