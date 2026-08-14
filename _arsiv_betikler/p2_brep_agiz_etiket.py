# -*- coding: utf-8 -*-
"""P2 OTOMATIK: mouth etiketini B-rep'in GERCEK SINIRINDAN uret (insan gerekmez).

WHY ONCEKI OTO-ETIKET COKTU: `g5_mouth_label` GT noktasi etrafina SABIT
yaricapli disk boyuyordu. Olculdu ki that yolla egitilen g10, gorulmemis markada
DAHA AZ candidate uretiyor (pool recall 0.6654 -> 0.5847,
[[g10-zinciri-d7de-gerileme]]). Once h3 de r=2mm diskle atesleme oranini
%78'den %24'e cokertmisti ([[h3-highcp-finetune-dead]]).

FARK: residual agzin GERCEK SINIRI elimizde. B-rep silindir agzi own YARICAPINI
tasiyor, duzlemsel opening own halkasini. Etiket sabit a diskle not,
parcanin own geometrisiyle ciziliyor -- "metrik-hizali mouth segmentasyonu"nun
full tanimi budur.

YONTEM (part basina):
 1. Uretici GT'si (konum + direction) with B-rep agizlarini ESLESTIR: GT'ye `ESLES_MM`
    inside and ekseni GT yonuyle `ESLES_ACI` inside which is most yakin mouth.
 2. Eslesen agzin GERCEK disk bolgesini boya: eksene dik uzaklik <= R*`R_PAY`,
    mouth duzleminden axial depth [-`GERI`*R, +`ILERI`*R].
 3. Hicbir GT eslesmiyorsa parcayi HIC YAZMA (wrong label bosluktan kotudur).

CIKTI SOZLESMESI (`train_seg_extra --partial-dir`): CableEntry=3 isaretlenir,
gerisi 0; maskeli loss only CE-vs-not'i denetler, unsigned sinifları
Housing diye OGRETMEZ.

TEZE SADIK: 5 sinif, ~6000 uniform izotropik remesh, `v_o` turetmesi DEGISMEZ.
Degisen single sey ETIKETIN AGZA NE KADAR OTURDUGU.
"""
import argparse 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
os .environ .setdefault ("BA_ALLOW_SEEN","1")

CE =3 
# ESLESME OLCUSU METRIGIN KENDISIYLE AYNI OLMALI.
# ILK SURUM YANLISTI: GT with mouth merkezi arasi DUZ OKLID mesafesi kullaniyordum
# and eslesme %19.5'te kaldi. Sebep: manufacturer CP'lerinin approximately yarisi GOVDE
# ICINDE duruyor ([[fiziksel-cerrahi-s-serisi]]), i.e. agizdan EKSENEL as
# uzakta. Urunun metrigi already axial kaymayi 40mm'ye up to serbest birakiyor
# and only YANAL hataya bakiyor. Etiket eslesmesi de oyle must be.
ESLES_YANAL =2.5 # GT'nin mouth EKSENINE dik uzakligi (mm)
ESLES_EKSENEL =40.0 # axis along serbest offset (mm) -- metrikle AYNI
ESLES_ACI =25.0 # GT yonu with mouth ekseni arasi most large angle
R_PAY =1.15 # mouth yaricapina pay (remesh vertex araligi for)
GERI ,ILERI =0.6 ,1.2 # mouth duzleminden geri/ileri axial band (R kati)
EN_AZ_TEPE =4 # this up to vertex boyanmayan mouth SAYILMAZ


def _birim (v ):
    n =np .linalg .norm (v )
    return v /n if n >1e-9 else None 


def agizlar (cyl ,acik ):
    """(centre, axis, radius) listesi -- silindir agizlari + duzlemsel acikliklar."""
    out =[]
    for c in cyl or []:
        a =_birim (np .asarray (c ["axis"],float ))
        if a is None :
            continue 
        r =float (c .get ("radius",0.0 ))
        if r <=1e-6 :
            continue 
        for u ,s in ((c .get ("mouth_a"),1.0 ),(c .get ("mouth_b"),-1.0 )):
            if u is not None :
                out .append ((np .asarray (u ,float ),s *a ,r ))
    for o in acik or []:
        if not isinstance (o ,dict )or o .get ("center")is None :
            continue 
        n =_birim (np .asarray (o .get ("normal",[0 ,0 ,1 ]),float ))
        r =float (o .get ("esd_r",0.0 ))
        if n is None or r <=1e-6 :
            continue 
        out .append ((np .asarray (o ["center"],float ),n ,r ))
    return out 


def boya (V ,G ,Gd ,cyl ,acik ):
    """Doner: (label dizisi, eslesen GT count, total GT)."""
    L =np .zeros (len (V ),np .int64 )
    A =agizlar (cyl ,acik )
    if not A :
        return L ,0 ,len (G )
    M =np .asarray ([a [0 ]for a in A ],float )
    AX =np .asarray ([a [1 ]for a in A ],float )
    eslesen =0 
    for j in range (len (G )):
        u =_birim (np .asarray (Gd [j ],float ))
        if u is None :
            continue 
            # YANAL/EKSENEL AYRIMI: GT'yi each agzin KENDI ekseni along ayristir.
        w =G [j ][None ]-M 
        eks =np .einsum ("ij,ij->i",w ,AX )
        yan =np .linalg .norm (w -eks [:,None ]*AX ,axis =1 )
        aci =np .degrees (np .arccos (
        np .clip (np .abs (AX @u ),-1.0 ,1.0 )))
        candidate =np .where ((yan <=ESLES_YANAL )&
        (np .abs (eks )<=ESLES_EKSENEL )&
        (aci <=ESLES_ACI ))[0 ]
        if not len (candidate ):
            continue 
            # most iyi = lateral hatasi most small which is (metrigin baktigi buyukluk)
        i =int (candidate [np .argmin (yan [candidate ])])
        m ,a ,r =A [i ]
        w =V -m 
        eks =w @a 
        dik =np .linalg .norm (w -eks [:,None ]*a [None ],axis =1 )
        k =(dik <=r *R_PAY )&(eks >=-GERI *r )&(eks <=ILERI *r )
        if int (k .sum ())<EN_AZ_TEPE :
            continue 
        L [k ]=CE 
        eslesen +=1 
    return L ,eslesen ,len (G )


def obj_yaz (yol ,V ,F ):
    with open (yol ,"w")as f :
        for v in V :
            f .write (f"v {v [0 ]:.6f} {v [1 ]:.6f} {v [2 ]:.6f}\n")
        for t in F :
            f .write (f"f {t [0 ]+1 } {t [1 ]+1 } {t [2 ]+1 }\n")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--output",default ="_p2_brep_etiket")
    ap .add_argument ("--most-few-ratio",type =float ,default =0.5 ,
    help ="parcanin GT'lerinin at least this orani eslesmeli")
    a =ap .parse_args ()
    import d6_record 
    import canonical_d7 as K 

    os .makedirs (a .out_ ,exist_ok =True )
    d7 ={str (p )for p in json .load (
    open ("results/d7_sinav_kumesi.json"))["pidler"]}

    isler =[
    ("corpus","results/_p1_olasilik_brepegit",
    "results/_brepegit_silindirler.pkl","results/_brepegit_acikliklar.pkl",
    K .yukle ([str (p )for p in json .load (
    open ("results/brep_egitim_kumesi.json"))["pidler"]])),
    ("d6","results/_p1_olasilik_g7","results/_d6_silindirler.pkl",
    "results/_d6_acikliklar.pkl",
    d6_record .yukle (set (d6_record .exam ()["pidler"]))),
    ]
    top ={"yazilan":0 ,"atlanan":0 ,"gt":0 ,"eslesen":0 }
    t0 =time .time ()
    for ad ,ob ,cylf ,acf ,kay in isler :
        cy =pickle .load (open (cylf ,"rb"))
        ac =pickle .load (open (acf ,"rb"))
        n =0 
        for pid ,r in sorted (kay .items ()):
            pid =str (pid )
            if pid in d7 :
                raise SystemExit (f"SIZINTI: {pid } D7'de -- training etiketi URETILMEZ")
            G =np .asarray (r .get ("G",[]),float )
            f =f"{ob }/{pid }.npz"
            if not len (G )or not os .path .exists (f ):
                top ["atlanan"]+=1 
                continue 
            z =np .load (f )
            V =np .asarray (z ["V"],float )
            F =np .asarray (z ["F"],np .int64 )
            L ,es ,ng =boya (V ,G ,np .asarray (r ["Gd"],float ),
            cy .get (pid ),ac .get (pid ))
            top ["gt"]+=ng 
            top ["eslesen"]+=es 
            if ng ==0 or es /ng <a .en_az_oran :
                top ["atlanan"]+=1 
                continue 
            d =os .path .join (a .out_ ,pid )
            os .makedirs (d ,exist_ok =True )
            obj_yaz (os .path .join (d ,f"{pid }.obj"),V ,F )
            np .savetxt (os .path .join (d ,f"{pid }.labels.txt"),L ,fmt ="%d")
            top ["yazilan"]+=1 
            n +=1 
            if n %200 ==0 :
                print (f"  {ad } {n } yazildi ({time .time ()-t0 :.0f}s)",flush =True )
        print (f"{ad } bitti: {n } part",flush =True )

    print (f"\nYAZILAN {top ['yazilan']} | ATLANAN {top ['atlanan']}")
    print (f"GT eslesme: {top ['eslesen']}/{top ['gt']} "
    f"(%{100 *top ['eslesen']/max (top ['gt'],1 ):.1f})")
    json .dump ({"yazilan":top ["yazilan"],"atlanan":top ["atlanan"],
    "gt":top ["gt"],"eslesen":top ["eslesen"],
    "esles_yanal":ESLES_YANAL ,"esles_eksenel":ESLES_EKSENEL ,
    "esles_aci":ESLES_ACI ,"r_pay":R_PAY ,
    "band":[GERI ,ILERI ],"en_az_oran":a .en_az_oran ,
    "not":"B-rep GERCEK mouth sinirindan uretilen kismi CableEntry "
    "etiketi. D7 kesisimi SIFIR (kod inside assert)."},
    open ("results/p2_brep_etiket.json","w"),indent =1 )
    print (f"-> {a .out_ }")


if __name__ =="__main__":
    main ()
