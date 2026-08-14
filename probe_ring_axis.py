# -*- coding: utf-8 -*-
"""HALKA NORMALI = EKSEN mi? (cevrimdisi, new inference YOK)

Simdiye up to halka normali only **ISARET** duzeltmek for kullanildi
(`halka_disari`, +0.0195 tanidik markada, KESIN). Ama fiziksel as
duz a yuzeydeki kablo girisinin EKSENI, that yuzeyin normalidir. Yani
halka normali only isareti not, **ekseni de** verebilir.

Bu, sign kolundan FARKLI a iddiadir and simdiye up to NOT MEASURED:
sign kollari `unsigned` metrigi never degistirmez (tanim geregi), axis
degistirmek whereas **hem unsigned hem signed** metrigi changes.

UC VARYANT:
  * `full`      : yonu tamamen halka normaliyle degistir
  * `kapili_A` : only prediction with halka normali arasindaki angle A
                 dereceden BUYUKSE degistir (tahmine guvenmedigimiz yer)
  * `harman_w` : w*halka + (1-w)*prediction, after normalize

Karar ESLI PARCA BOOTSTRAP with verilir.
"""
import json 
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from sina_cluster import match_hungarian # noqa: E402

DOKUM =os .environ .get ("HE_DOKUM","results/_dokum_halka.json")
YOL =os .environ .get ("HE_YOL","saha")


def _birim (v ):
    v =np .asarray (v ,float ).reshape (-1 ,3 )
    return v /np .maximum (np .linalg .norm (v ,axis =1 ,keepdims =True ),1e-12 )


def uygula (D ,H ,arm ):
    """D: prediction yonleri (n,3). H: halka normalleri (n,3), sifir = absent."""
    if not len (D ):
        return D 
    gecerli =np .linalg .norm (H ,axis =1 )>1e-9 
    Y =D .copy ()
    if arm =="baseline":
        return Y 
    if arm =="sign":# mevcut, dagitima candidate rule
        c =np .sum (D *H ,axis =1 )
        cevir =gecerli &(c <0 )
        Y [cevir ]=-D [cevir ]
        return Y 
    if arm =="tam":
        Y [gecerli ]=H [gecerli ]
        return Y 
    if arm .startswith ("kapili_"):
        A =float (arm .split ("_")[1 ])
        # before isareti hizala, after KALAN angle farkina bak
        c =np .sum (D *H ,axis =1 )
        Hs =H *np .where (c <0 ,-1.0 ,1.0 )[:,None ]# H'yi D'ye yaklastir
        aci =np .degrees (np .arccos (np .clip (np .sum (D *Hs ,axis =1 ),-1 ,1 )))
        deg =gecerli &(aci >A )
        Y [deg ]=H [deg ]# ISARETIYLE birlikte halka
        # kalanlarda only sign duzeltmesi
        kal =gecerli &~deg &(c <0 )
        Y [kal ]=-D [kal ]
        return Y 
    if arm .startswith ("harman_"):
        w =float (arm .split ("_")[1 ])
        c =np .sum (D *H ,axis =1 )
        Ds =D *np .where (c <0 ,-1.0 ,1.0 )[:,None ]# D'yi H isaretine al
        v =w *H +(1 -w )*Ds 
        n =np .linalg .norm (v ,axis =1 )
        ok =gecerli &(n >1e-9 )
        Y [ok ]=v [ok ]/n [ok ][:,None ]
        return Y 
    raise ValueError (arm )


def olc (kayit ,arm ):
    tot ={k :[0 ,0 ,0 ]for k in ("tespit","rob","rbi")}
    part =[]
    for r in kayit :
        G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
        if not len (G ):
            continue 
        Gd =_birim (r ["Gd"])
        P =np .asarray (r ["P"],float ).reshape (-1 ,3 )
        D =_birim (r ["D"])if len (P )else np .zeros ((0 ,3 ))
        H =np .asarray (r .get ("halka_normal")or [],float ).reshape (-1 ,3 )
        if len (H )!=len (D ):
            H =np .zeros_like (D )
        D2 =uygula (D ,_birim (H )if len (H )else H ,arm )
        satir ={}
        for ad ,(tol ,am ,isr )in (("tespit",(2.0 ,180.0 ,False )),
        ("rob",(2.0 ,10.0 ,False )),
        ("rbi",(2.0 ,10.0 ,True ))):
            tp ,fp ,fn ,_ =match_hungarian (P ,D2 ,G ,Gd ,float (r ["diag"]),
            tol ,am ,False ,signed =isr )
            tot [ad ][0 ]+=tp 
            tot [ad ][1 ]+=fp 
            tot [ad ][2 ]+=fn 
            satir [ad ]=(tp ,fp ,fn )
        part .append (satir )
    f1 =lambda t :2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )# noqa: E731
    return {k :f1 (v )for k ,v in tot .items ()},part 


def boot (pa ,pb ,ad ,n =4000 ,seed =0 ):
    rng =np .random .default_rng (seed )
    A =np .asarray ([r [ad ]for r in pa ],float )
    B =np .asarray ([r [ad ]for r in pb ],float )
    f1 =lambda t :2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )# noqa: E731
    d =[]
    for _ in range (n ):
        i =rng .integers (0 ,len (A ),len (A ))
        d .append (f1 (B [i ].sum (0 ))-f1 (A [i ].sum (0 )))
    d =np .asarray (d )
    return d .mean (),np .percentile (d ,2.5 ),np .percentile (d ,97.5 ),(d >0 ).mean ()


def main ():
    kayit =[r for r in json .load (open (DOKUM ))if r .get ("yol")==YOL 
    and r .get ("halka_normal")]
    print (f"{DOKUM } / yol={YOL } -> {len (kayit )} part (halka normali olan)")
    if not kayit :
        return 1 
    KOLLAR =["baseline","sign","tam",
    "kapili_10","kapili_20","kapili_30","kapili_45",
    "harman_0.25","harman_0.5","harman_0.75"]
    sonuc ={}
    print (f"\n{'arm':14s} {'tespit':>8s} {'rob':>8s} {'rob-ISR':>8s}")
    for arm in KOLLAR :
        m ,p =olc (kayit ,arm )
        sonuc [arm ]=(m ,p )
        print (f"{arm :14s} {m ['tespit']:8.4f} {m ['rob']:8.4f} {m ['rbi']:8.4f}")
    print ("\n--- ESLI BOOTSTRAP (tabana gore) ---")
    print (f"{'arm':14s} {'metrik':>7s} {'fark':>9s} {'%95 GA':>22s} {'poz%':>6s}")
    for arm in KOLLAR [1 :]:
        for ad in ("rob","rbi"):
            f ,lo ,hi ,pz =boot (sonuc ["baseline"][1 ],sonuc [arm ][1 ],ad )
            yz =" *"if (lo >0 or hi <0 )else ""
            print (f"{arm :14s} {ad :>7s} {f :+9.4f} "
            f"[{lo :+.4f},{hi :+.4f}]{yz :>3s} {100 *pz :5.1f}")
    json .dump ({k :v [0 ]for k ,v in sonuc .items ()},
    open (f"results/halka_eksen_{YOL }.json","w"),indent =1 )
    print (f"\n-> results/halka_eksen_{YOL }.json")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
