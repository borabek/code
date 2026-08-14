# -*- coding: utf-8 -*-
"""VII.0c-g -- KAFES ARAMASI, GT'SIZ OLCUTLE (uretimde yapilabilir haliyle)

VII.0b'DEKI DEFECT. Kafesi ararken hedef fonksiyon as **GT kapsamasini**
kullanmistim. Yani arama KAHIN by yonlendiriliyordu; uretimde boyle
a sey yapilamaz. O yuzden NIT'in 0.323'u "arama zayif" not, more kotusu:
kahin yardimiyla bile skorla filtrelenmis candidate bulutundan iyi lattice cikmiyor.

BU SURUMDE UC SEY DEGISTI:
  1. ARAMA OLCUTU GT'SIZ: lattice, uzerine dusen ADAY count and izgara
     DOLULUGU with puanlanir. GT only DEGERLENDIRMEDE is used.
  2. TAM HAVUZ: candidates skora according to filtrelenmez. Kafes GEOMETRIK a
     ozelliktir; NIT'te GT'nin only %5.3'u first-k inside oldugu for skor
     filtresi kafesi YOK EDIYORDU.
  3. ADIL TOLERANS + DOYGUNLUK: kahinle same tolerans, and 6 kafese up to
     (NIT'te kapsama 3'te still tirmaniyordu: 0.161/0.255/0.323).

Model EGITIMI YOK -- only candidate geometrisi and GT (degerlendirme for).
D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
from run_p6_ortak import yukle # noqa: E402

KUME =os .environ .get ("KV_KUME","d6")
TOL =float (os .environ .get ("KV_TOL","1.0"))# kahinle AYNI
N_KAFES =int (os .environ .get ("KV_N","6"))
MIN_ADIM =float (os .environ .get ("KV_MIN_ADIM","2.0"))
MARKALAR =set (os .environ .get ("KV_MARKA","NIT,MOR,SUPU,UPUN").split (","))
# KAYNAK: "candidate" = mesh+brep candidate bulutu | "brep" = YALNIZ B-rep silindir
# merkezleri. Mesh tepeleri ~6000'e DUZGUN yeniden orneklendigi for KENDILERI
# periyodiktir; GT'siz arama CP kafesini not MESH'IN ORNEKLEME kafesini
# buluyor (measured: kapsama 0.001-0.115). B-rep merkezleri analitik and few.
KAYNAK =os .environ .get ("KV_KAYNAK","candidate")


def _izgara (P ,seed ,step_ ,n_max =80 ):
    L =float (np .linalg .norm (step_ ))
    if L <MIN_ADIM :
        return np .zeros ((0 ,3 )),0 
    u =step_ /L 
    t =(P -seed )@u 
    n0 =max (int (np .floor (t .min ()/L ))-1 ,-n_max )
    n1 =min (int (np .ceil (t .max ()/L ))+1 ,n_max )
    ns =np .arange (n0 ,n1 +1 )
    return seed [None ,:]+ns [:,None ]*(u *L )[None ,:],len (ns )


def _yakin (A ,B ,tol ):
    """A'nin each satiri B'ye tol inside mi (OKLIT) -- ADAY tarafi for."""
    if not len (A )or not len (B ):
        return np .zeros (len (A ),bool )
    return (np .linalg .norm (A [:,None ,:]-B [None ,:,:],axis =-1 ).min (1 )
    <=tol )


def _gt_kapsandi (G ,Gd ,uretilen ,lateral =2.0 ,axial =40.0 ):
    """GT, uretilen izgara noktalarindan biriyle URUN KABUL KUTUSUNDA mi?

    IMPORTANT DUZELTME (2026-08-12): butun lattice sondalarini OKLIT 1mm with
    degerlendirmistim. Urun metrigi whereas YANAL <= 2mm, EKSENEL <= 40mm --
    i.e. GT yonunde 10mm kaymis a point KABUL EDILIR. B-rep silindir
    merkezi/agzi full da eksende kayik oldugu for (measured: GT'ye oklit
    median 2.4-12mm) Oklit criterion kolu HAKSIZ YERE olu gosteriyordu.
    """
    if not len (G )or not len (uretilen ):
        return np .zeros (len (G ),bool )
    Gn =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
    v =uretilen [:,None ,:]-G [None ,:,:]# (uret, gt, 3)
    al =(v *Gn [None ,:,:]).sum (-1 )
    yan =np .linalg .norm (v -al [...,None ]*Gn [None ,:,:],axis =-1 )
    return ((yan <=lateral )&(np .abs (al )<=axial )).any (0 )


def kafes_ara (P ,n_kafes =N_KAFES ,rng =None ):
    """GT'SIZ arama: izgarayi 'uzerine dusen ADAY count x DOLULUK' with puanla.

    DOLULUK = isabet / izgara noktasi. Bu olmadan arama COK KUCUK adimi selects
    (each seyi kapsar but anlamsizdir); MIN_ADIM with birlikte two yonlu koruma.
    """
    rng =rng or np .random .default_rng (0 )
    kalan_aday =np .ones (len (P ),bool )
    bulunan =[]
    if len (P )<3 :
        return bulunan 
    for _ in range (n_kafes ):
        idx =np .where (kalan_aday )[0 ]
        if len (idx )<3 :
            break 
        Pk =P [idx ]
        farklar =(Pk [:,None ,:]-Pk [None ,:,:]).reshape (-1 ,3 )
        boy =np .linalg .norm (farklar ,axis =1 )
        iyi =boy >=MIN_ADIM 
        if not iyi .any ():
            break 
        candidate =farklar [iyi &(boy <=np .percentile (boy [iyi ],60 ))]
        if not len (candidate ):
            break 
        if len (candidate )>200 :
            candidate =candidate [rng .choice (len (candidate ),200 ,replace =False )]
        candidate =np .vstack ([candidate ,candidate /2.0 ,candidate /3.0 ])# lower harmonikler
        tohumlar =Pk [::max (1 ,len (Pk )//12 )]
        en =(None ,-1.0 ,None )
        for step_ in candidate :
            if np .linalg .norm (step_ )<MIN_ADIM :
                continue 
            for seed in tohumlar :
                uret ,n_gr =_izgara (P ,seed ,step_ )
                if n_gr <3 :
                    continue 
                isabet =_yakin (uret ,P ,TOL )
                n_hit =int (isabet .sum ())
                if n_hit <3 :
                    continue 
                doluluk =n_hit /max (n_gr ,1 )
                score_ =n_hit *doluluk # GT KULLANILMIYOR
                if score_ >en [1 ]:
                    en =((seed ,step_ ),score_ ,uret )
        if en [0 ]is None :
            break 
        bulunan .append (en )
        # this kafesin uzerine dusen ADAYLARI cikar
        kapsanan =_yakin (P ,en [2 ],TOL )
        kalan_aday =kalan_aday &~kapsanan 
    return bulunan 


_CY ={}


def main ():
    t0 =time .time ()
    if KAYNAK =="brep":
        import pickle 
        global _CY 
        _CY =pickle .load (open (os .environ .get (
        "KV_SIL","results/_d6_silindirler.pkl"),"rb"))
        print (f"B-rep silindir onbellegi: {len (_CY )} part",flush =True )
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for d in data_ :
        if d ["mfg"]not in MARKALAR :
            continue 
        G =np .asarray (d ["G"],float )
        if len (G )<3 :
            continue 
        if KAYNAK =="brep":
            sil =_CY .get (str (d ["pid"]))or []
            if len (sil )<3 :
                continue 
            P =np .unique (np .round (
            np .array ([np .asarray (c ["center"],float )for c in sil ]),3 ),
            axis =0 )
        else :
            P =np .unique (np .round (np .asarray (d ["P"],float ),3 ),axis =0 )
        bulunan =kafes_ara (P )
        a =ist [d ["mfg"]]
        a ["gt"].append (len (G ))
        a ["candidate"].append (len (P ))
        Gd =np .asarray (d ["Gd"],float )
        kalan =np .ones (len (G ),bool )
        for i in range (N_KAFES ):
            if i <len (bulunan ):
                kalan =kalan &~_gt_kapsandi (G ,Gd ,bulunan [i ][2 ])
            a [f"k{i +1 }"].append (int ((~kalan ).sum ()))
        if bulunan :
            a ["adim1"].append (float (np .linalg .norm (bulunan [0 ][0 ][1 ])))
        n +=1 
        if n %40 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    oracle_ ={"NIT":0.983 ,"SUPU":0.864 ,"MOR":0.811 ,"UPUN":0.810 }
    bas ="".join (f"{i }k".rjust (7 )for i in range (1 ,N_KAFES +1 ))
    print (f"\n{'brand':<7}{'GT':>6}{'candidate':>6}{bas }{'KAHIN':>8}{'acik':>7}"
    f"{'adim1':>8}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =sum (a ["gt"])
        r ={"gt":g ,"aday_parca":float (np .mean (a ["candidate"]))}
        sat =""
        for i in range (1 ,N_KAFES +1 ):
            v =sum (a [f"k{i }"])/max (g ,1 )
            r [f"lattice{i }"]=v 
            sat +=f"{v :>7.3f}"
        kh =oracle_ .get (m_ ,0.0 )
        r ["kahin"]=kh 
        r ["acik"]=kh -r [f"lattice{N_KAFES }"]
        r ["adim1_ortanca"]=float (np .median (a ["adim1"]))if a ["adim1"]else 0.0 
        out [m_ ]=r 
        print (f"{m_ :<7}{g :>6}{r ['aday_parca']:>6.0f}{sat }{kh :>8.3f}"
        f"{r ['acik']:>7.3f}{r ['adim1_ortanca']:>8.2f}")

    json .dump ({"tol":TOL ,"n_kafes":N_KAFES ,"min_adim":MIN_ADIM ,
    "brand":out ,
    "not":"ARAMA GT'SIZ: izgara 'isabet x doluluk' with puanlanir, "
    "GT only DEGERLENDIRMEDE. Adaylar skorla "
    "FILTRELENMEZ (lattice geometrik a ozelliktir). "
    "D7'ye BAKILMADI."},
    open (f"results/kafes_v2_{KUME }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/kafes_v2_{KUME }.json")


if __name__ =="__main__":
    main ()
