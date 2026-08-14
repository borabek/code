# -*- coding: utf-8 -*-
"""VII.0 -- COKLU KAFES TAVANI: 0.70 mumkun mu?

TEK OTELEME sondasi NIT for tekrar tavanini **0.557** olctu. 0.70 hedefi whereas
NIT'te 0.66 istiyor, i.e. single-oteleme tavaninin (%92'si) neredeyse tamamini
and HIC wrong pozitif olmamasini. Gercekci not.

AMA klemenslerde most zaman BIRDEN COK array vardir (two order, two fold, ten/arka
face). Tavan single otelemeyle sinirli DEGIL. Bu probe acgozlu as birden very
lattice removes and KUMULATIF kapsamayi olcer:

    1 lattice -> ? | 2 lattice -> ? | 3 lattice -> ?

EGER 3 kafesle NIT 0.80+'a cikiyorsa 0.70 KONUSULABILIR hale gelir.
Cikmiyorsa 0.70 yapisal as closed demektir and bunu simdi bilmek iyidir.

AYRICA HARMONIK TUZAGI SINANIR: found step L for L/2 and L/3 de denenir.
Onceki probe NIT'te 10.50mm buldu, oysa standart klemens adimlari 3.5-7.5mm --
i.e. muhtemelen 2x harmonik bulunuyor and URETILEN IZGARA HER IKINCI CP'YI
ATLIYOR. Alt harmonik more very GT tutuyorsa ceiling YUKSELIR.

MODEL EGITIMI YOK -- only GT geometrisi. Hizli.
D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402

TOL =float (os .environ .get ("CK_TOL","1.0"))# izgara-GT esleme toleransi
N_KAFES =int (os .environ .get ("CK_N","3"))
KATALOG =(3.5 ,3.81 ,5.0 ,5.08 ,6.2 ,7.5 ,7.62 ,10.16 ,12.7 )


def _izgara_tut (G ,seed ,step_ ):
    """`seed + n*step` izgarasina TOL inside dusen GT maskesi."""
    L =float (np .linalg .norm (step_ ))
    if L <1e-6 :
        return np .zeros (len (G ),bool )
    u =step_ /L 
    v =G -seed 
    t =v @u 
    dik =np .linalg .norm (v -t [:,None ]*u [None ,:],axis =1 )
    n =np .round (t /L )
    error =np .abs (t -n *L )
    return (dik <=TOL )&(error <=TOL )


def _en_iyi_kafes (G ,remaining ):
    """Kalan GT'leri EN COK kapsayan (seed, step) ikilisi."""
    idx =np .where (remaining )[0 ]
    if len (idx )<3 :
        return None ,0 ,0.0 
    Gk =G [idx ]
    farklar =(Gk [:,None ,:]-Gk [None ,:,:]).reshape (-1 ,3 )
    boy =np .linalg .norm (farklar ,axis =1 )
    iyi =(boy >0.5 )&(boy <np .percentile (boy [boy >0.5 ],50 ))
    candidate =farklar [iyi ]
    if not len (candidate ):
        return None ,0 ,0.0 
    if len (candidate )>300 :
        candidate =candidate [np .random .default_rng (0 ).choice (len (candidate ),300 ,False )]
        # ALT HARMONIKLER de candidate: L, L/2, L/3
    genis =[candidate ]
    for b in (2.0 ,3.0 ):
        genis .append (candidate /b )
    candidate =np .vstack (genis )
    en_maske ,en_n ,en_L =None ,0 ,0.0 
    for step_ in candidate :
        for seed in Gk [::max (1 ,len (Gk )//8 )]:
            m =_izgara_tut (G ,seed ,step_ )&remaining 
            n =int (m .sum ())
            if n >en_n :
                en_maske ,en_n ,en_L =m ,n ,float (np .linalg .norm (step_ ))
    return en_maske ,en_n ,en_L 


def coklu (G ,n_kafes =N_KAFES ):
    """Acgozlu: most iyi kafesi bul, kapsananlari cikar, tekrarla."""
    G =np .asarray (G ,float )
    remaining =np .ones (len (G ),bool )
    kapsam ,adimlar =[],[]
    for _ in range (n_kafes ):
        m ,n ,L =_en_iyi_kafes (G ,remaining )
        if m is None or n <=1 :
            break 
        remaining =remaining &~m 
        kapsam .append (1.0 -remaining .mean ())
        adimlar .append (round (L ,2 ))
    while len (kapsam )<n_kafes :
        kapsam .append (kapsam [-1 ]if kapsam else 0.0 )
    return kapsam ,adimlar 


def main ():
    t0 =time .time ()
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    hedef =set (os .environ .get ("CK_MARKA","NIT,MOR,SUPU,UPUN").split (","))
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for pid ,r in kay .items ():
        mfg =r .get ("mfg")
        if mfg not in hedef :
            continue 
        G =np .asarray (r .get ("G",[]),float )
        if len (G )<3 :
            continue 
        kaps ,step_ =coklu (G )
        a =ist [mfg ]
        a ["gt"].append (len (G ))
        for i ,k in enumerate (kaps ):
            a [f"lattice{i +1 }"].append (k *len (G ))
        if step_ :
            a ["adim1"].append (step_ [0 ])
        n +=1 
        if n %50 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    print (f"\n{'brand':<7}{'part':>6}{'GT':>7}{'1 lattice':>9}{'2 lattice':>9}"
    f"{'3 lattice':>9}{'adim1':>8}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =sum (a ["gt"])
        r ={"part":len (a ["gt"]),"gt":g }
        for i in (1 ,2 ,3 ):
            r [f"lattice{i }"]=sum (a [f"lattice{i }"])/max (g ,1 )
        r ["adim1_ortanca"]=float (np .median (a ["adim1"]))if a ["adim1"]else 0.0 
        out [m_ ]=r 
        print (f"{m_ :<7}{r ['part']:>6}{g :>7}{r ['kafes1']:>9.3f}"
        f"{r ['kafes2']:>9.3f}{r ['kafes3']:>9.3f}"
        f"{r ['adim1_ortanca']:>8.2f}")

        # 0.70 KARARI
    nit =out .get ("NIT",{})
    k3 =nit .get ("kafes3",0.0 )
    f1_tav =2 *k3 /(1 +k3 )if k3 else 0.0 
    print (f"\n0.70 KARARI -- NIT 3 kafeste kapsama {k3 :.3f} "
    f"-> F1 tavani {f1_tav :.4f}")
    print (f"  0.70 for NIT'te ~0.66 gerekiyordu.")
    print (f"  {'KONUSULABILIR'if f1_tav >=0.75 else 'YAPISAL OLARAK ZOR'}"
    f" (ceiling {f1_tav :.3f})")
    json .dump ({"tol":TOL ,"n_kafes":N_KAFES ,"brand":out ,
    "nit_3kafes_f1_tavani":f1_tav ,
    "not":"Acgozlu coklu lattice: en iyi izgarayi bul, kapsananlari "
    "cikar, tekrarla. Alt harmonikler (L/2, L/3) de candidate. "
    "MODEL YOK -- yalnizca GT geometrisi. D7'ye BAKILMADI."},
    open ("results/coklu_lattice.json","w"),indent =1 )
    print ("receipt -> results/coklu_lattice.json")


if __name__ =="__main__":
    main ()
