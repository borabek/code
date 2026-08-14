# -*- coding: utf-8 -*-
"""POSE KIRPMA SINIRI (`maks_mm`) TARAMASI -- cevrimdisi, single kosudan.

RATIONALE (2026-08-14 teshisi, Bolum 21.46): baglayici kisit YANAL KONUM.
GT'lerin **%83'u** for havuzda, yonu already 10 derece inside correct which is
a candidate **10 mm** yakinda duruyor; but pose head'in duzeltmesi
**3 mm**'ye kirpiliyor (`pose_head.pkl: maks_mm=3.0`). Bu hiperparametrenin
tarandigina dair kayit YOK.

Sonda, kirpma ONCESI yer degistirme vektorunu dokuyor (`pose_dw`), this
yuzden TEK kosudan each `maks_mm` degeri cevrimdisi yeniden kurulabilir.

VERIFICATION: mx = 3.0'da yeniden kurulan metrik, dokumun own metrigine
ESIT must be. Esit degilse tarama gecersizdir and betik bunu SOYLER.
"""
import json 
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from sina_cluster import match_hungarian # noqa: E402

DOKUM =os .environ .get ("PK_DOKUM","results/_dump_pose.json")
YOL =os .environ .get ("PK_YOL","field")
MEVCUT =3.0 


def _birim (v ):
    v =np .asarray (v ,float ).reshape (-1 ,3 )
    return v /np .maximum (np .linalg .norm (v ,axis =1 ,keepdims =True ),1e-12 )


def kur (r ,mx ,guven_esik =None ,mx_dusuk =MEVCUT ):
    """kirpma siniri mx with noktalari yeniden kur.

    `guven_esik` verilirse KOSULLU kirpma: guveni esigin ustunde which is
    adaylarda boundary `mx`, digerlerinde `mx_dusuk` (mevcut 3 mm). Gerekce:
    large duzeltmeye however modelin emin oldugu places izin vermek, "emin
    degilse dokunma" ilkesinin this koldaki karsiligidir.
    """
    p0 =np .asarray (r .get ("pose_p0")or [],float ).reshape (-1 ,3 )
    dw =np .asarray (r .get ("pose_dw")or [],float ).reshape (-1 ,3 )
    P =np .asarray (r ["P"],float ).reshape (-1 ,3 )
    D =_birim (r ["D"])if len (P )else np .zeros ((0 ,3 ))
    if not len (p0 )or len (p0 )!=len (P ):
    # pose kancasi with output hizalanmiyor (pose sonrasi eleme olmus)
        return None ,None 
    n =np .linalg .norm (dw ,axis =1 )
    if guven_esik is None :
        bound_ =np .full (len (n ),float (mx ))
    else :
        g =np .asarray (r .get ("confidence")or [],float ).reshape (-1 )
        if len (g )!=len (n ):
            return None ,None 
        g =np .where (np .isfinite (g ),g ,-np .inf )
        bound_ =np .where (g >=guven_esik ,float (mx ),float (mx_dusuk ))
    ol =np .where (n >1e-9 ,np .minimum (n ,bound_ )/np .maximum (n ,1e-12 ),0.0 )
    return p0 +dw *ol [:,None ],D 


def olc (rec_ ,mx =None ,guven_esik =None ):
    tot ={k :[0 ,0 ,0 ]for k in ("detection","rob","rbi")}
    part ,skipped =[],0 
    for r in rec_ :
        G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
        if not len (G ):
            continue 
        Gd =_birim (r ["Gd"])
        if mx is None :
            P =np .asarray (r ["P"],float ).reshape (-1 ,3 )
            D =_birim (r ["D"])if len (P )else np .zeros ((0 ,3 ))
        else :
            P ,D =kur (r ,mx ,guven_esik )
            if P is None :
                skipped +=1 
                P =np .asarray (r ["P"],float ).reshape (-1 ,3 )
                D =_birim (r ["D"])if len (P )else np .zeros ((0 ,3 ))
        line_ ={}
        for ad ,(tol ,am ,isr )in (("detection",(2.0 ,180.0 ,False )),
        ("rob",(2.0 ,10.0 ,False )),
        ("rbi",(2.0 ,10.0 ,True ))):
            tp ,fp ,fn ,_ =match_hungarian (P ,D ,G ,Gd ,float (r ["diag"]),
            tol ,am ,False ,signed =isr )
            tot [ad ][0 ]+=tp 
            tot [ad ][1 ]+=fp 
            tot [ad ][2 ]+=fn 
            line_ [ad ]=(tp ,fp ,fn )
        part .append (line_ )
    f1 =lambda t :2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )# noqa: E731
    return {k :f1 (v )for k ,v in tot .items ()},part ,skipped 


def boot (pa ,pb ,ad ,n =4000 ,seed =0 ):
    rng =np .random .default_rng (seed )
    A =np .asarray ([r [ad ]for r in pa ],float )
    B =np .asarray ([r [ad ]for r in pb ],float )
    f1 =lambda t :2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )# noqa: E731
    d =[f1 (B [i ].sum (0 ))-f1 (A [i ].sum (0 ))
    for i in (rng .integers (0 ,len (A ),len (A ))for _ in range (n ))]
    d =np .asarray (d )
    return d .mean (),np .percentile (d ,2.5 ),np .percentile (d ,97.5 ),(d >0 ).mean ()


def main ():
    rec_ =[r for r in json .load (open (DOKUM ))if r .get ("path")==YOL ]
    print (f"{DOKUM } / path={YOL } -> {len (rec_ )} part")
    kanca_var =sum (1 for r in rec_ if r .get ("pose_dw"))
    print (f"pose kancasi which part: {kanca_var }")
    if not kanca_var :
        print ("POSE KANCASI BOS -- tarama yapilamaz")
        return 1 

    dok ,dok_p ,_ =olc (rec_ ,None )
    kur3 ,kur3_p ,atl =olc (rec_ ,MEVCUT )
    print (f"\nDOGRULAMA (mx={MEVCUT } yeniden kurulan == dokum?)  "
    f"hizalanmayan part: {atl }")
    print (f"  dokum       : detection {dok ['detection']:.4f} rob {dok ['rob']:.4f} "
    f"rbi {dok ['rbi']:.4f}")
    print (f"  yeniden kur : detection {kur3 ['detection']:.4f} "
    f"rob {kur3 ['rob']:.4f} rbi {kur3 ['rbi']:.4f}")
    ok =all (abs (dok [k ]-kur3 [k ])<1e-6 for k in dok )
    print (f"  -> {'GECERLI'if ok else 'UYUSMUYOR -- tarama SUPHELI'}")

    MX =[0.0 ,1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,8.0 ,10.0 ,15.0 ,1e9 ]
    res_ ={}
    print (f"\n{'maks_mm':>9s} {'detection':>8s} {'rob':>8s} {'rob-ISR':>8s}")
    for mx in MX :
        m ,p ,_ =olc (rec_ ,mx )
        res_ [mx ]=(m ,p )
        ad ="SINIRSIZ"if mx >1e8 else f"{mx :.1f}"
        yz ="  <- MEVCUT"if mx ==MEVCUT else ""
        print (f"{ad :>9s} {m ['detection']:8.4f} {m ['rob']:8.4f} "
        f"{m ['rbi']:8.4f}{yz }")

    print ("\n--- ESLI BOOTSTRAP (mevcut 3.0'a per) ---")
    print (f"{'maks_mm':>9s} {'metrik':>7s} {'diff':>9s} {'%95 GA':>22s} "
    f"{'poz%':>6s}")
    baseline =res_ [MEVCUT ][1 ]
    for mx in MX :
        if mx ==MEVCUT :
            continue 
        for ad in ("rob","rbi"):
            f ,lo ,hi ,pz =boot (baseline ,res_ [mx ][1 ],ad )
            yz =" *"if (lo >0 or hi <0 )else ""
            nm ="SINIRSIZ"if mx >1e8 else f"{mx :.1f}"
            print (f"{nm :>9s} {ad :>7s} {f :+9.4f} "
            f"[{lo :+.4f},{hi :+.4f}]{yz :>3s} {100 *pz :5.1f}")

            # GUVEN KAPILI KIRPMA: large duzeltmeye only emin oldugu places izin
    en_iyi =max ((k for k in MX if k <=1e8 ),
    key =lambda k :res_ [k ][0 ]["rbi"])
    if en_iyi !=MEVCUT :
        print (f"\n--- GUVEN KAPILI KIRPMA (emin ise {en_iyi :.1f} mm, "
        f"degilse {MEVCUT :.1f} mm) ---")
        print (f"{'threshold':>6s} {'detection':>8s} {'rob':>8s} {'rob-ISR':>8s} "
        f"{'diff(rbi)':>10s} {'%95 GA':>22s}")
        for threshold in (0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ):
            m ,p ,_ =olc (rec_ ,en_iyi ,threshold )
            f ,lo ,hi ,_ =boot (baseline ,p ,"rbi")
            yz =" *"if (lo >0 or hi <0 )else ""
            print (f"{threshold :6.2f} {m ['detection']:8.4f} {m ['rob']:8.4f} "
            f"{m ['rbi']:8.4f} {f :+10.4f} "
            f"[{lo :+.4f},{hi :+.4f}]{yz :>3s}")

            # yer degistirme buyuklugu dagilimi -- kirpmanin ne up to bagladigi
    n =np .concatenate ([np .linalg .norm (np .asarray (r ["pose_dw"],float )
    .reshape (-1 ,3 ),axis =1 )
    for r in rec_ if r .get ("pose_dw")])
    print (f"\nOnerilen yer degistirme (|dw|, {len (n )} CP): "
    f"ortanca {np .median (n ):.2f} mm, %75 {np .percentile (n ,75 ):.2f}, "
    f"%90 {np .percentile (n ,90 ):.2f}, maks {n .max ():.2f}")
    print (f"3 mm'yi ASAN oneri orani: {100 *(n >3 ).mean ():.1f}%")

    json .dump ({("SINIRSIZ"if k >1e8 else k ):v [0 ]
    for k ,v in res_ .items ()},
    open ("results/pose_clip.json","w"),indent =1 )
    print ("\n-> results/pose_clip.json")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
