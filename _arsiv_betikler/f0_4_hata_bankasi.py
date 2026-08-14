# -*- coding: utf-8 -*-
"""F0-4: HATA BANKASI -- each FP'ye FIZIKSEL label, each FN'e REASON etiketi.

WHY: F3 kollari and F1-8 "hangi hatayi olduruyorum" sorusunu yanitlayamadan kosuluyordu.
[[ablasyon-cephe-kararliligi]] and F0-5 hedef aritmetigi this bankayi input takes.

FN REASON ETIKETI -- three kova, ONBELLEKTEN conclusive ayrilabiliyor:
  ADAY_YOK    : GT'nin toleransinda HIC candidate absent            -> TURETME/COZUNURLUK sorunu
  GATE_REDDI  : candidate vardi but karar kurali reddetti        -> GATE sorunu
  ATAMA       : candidate vardi, kabul edildi, but BASKA GT'ye atandi -> METRIK/coklu esleme

Bu ayrim onemli because three kova UC AYRI kola bakar: ADAY_YOK -> F4 (temsil), GATE_REDDI ->
C/E kollari, ATAMA -> metrik. Havuzlanmis "366 FN" count hangisine bakacagini soylemiyor.

FP FIZIKSEL ETIKETI: A4 bayraklari (body ici / onu closed / mouth ic capi / ileri distance).
[[fizik-bayes-tavanini-oynatmiyor]]: bunlar TAVANI oynatmiyor but HATA TARIFI for gecerli --
"tavani oynatmiyor" with "hatayi tarif etmiyor" same sey not.

TEZ DEGISMEZ: no sey egitilmez, no threshold does not change. Bu a MUHASEBE betigi.
"""
import collections 
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
import wire_gate 

CIKTI ="results/f0_4_hata_bankasi.json"


def esle (P ,G ,Gd ,tol ):
    """Metrikle AYNI tekil eslesme. Doner: (candidate->gt sozlugu, eslesmeyen gt listesi)."""
    if not len (P )or not len (G ):
        return {},list (range (len (G )))
    d =P [:,None ,:]-G [None ,:,:]
    al =(d *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
    pe =np .where (np .abs (al )>40 ,np .inf ,pe )
    up ,ug ,cift =set (),set (),{}
    for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
    for a in range (len (P ))for b in range (len (G ))):
        if dd >tol or a_ in up or b_ in ug :
            continue 
        up .add (a_ );ug .add (b_ );cift [a_ ]=b_ 
    return cift ,[b for b in range (len (G ))if b not in ug ]


def main ():
    import protocol 
    protocol .tez_dogrula ()
    DER ,gate ,ek =T2 .yukle ()
    BAY ={}
    if os .path .exists ("results/_a4_bayraklar.pkl"):
        with open ("results/_a4_bayraklar.pkl","rb")as f :
            BAY =pickle .load (f )

    FP ,FN ,TP ,KAPSAMA =[],[],[],[]
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        M =np .hstack ([r ["X"],r ["XR"]])
        if M .shape [1 ]*2 !=gate ["n_feat"]:
            continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (gate ,M ))
        Ptum =np .asarray (r ["P"],float )# TUM candidates (gate ONCESI)
        P =Ptum [k ]# kabul edilenler
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        regime ="dusuk-CP"if len (G )<4 else "yuksek-CP"
        ortak ={"pid":r ["pid"],"mfg":r ["mfg"],"geo":r ["geo"],
        "regime":regime ,"n_gt":int (len (G ))}

        cift ,fn_idx =esle (P ,G ,Gd ,tol )
        bl =BAY .get (r ["pid"])or []
        for i in range (len (P )):
            kayit =dict (ortak )
            if len (bl )==len (P ):
                b =bl [i ]
                kayit ["govde_ici"]=bool (b ["govde_ici"])
                kayit ["ileri"]=(float (b ["ileri"])if np .isfinite (b ["ileri"])else None )
                kayit ["ic_cap"]=(float (b ["ic_cap"])if np .isfinite (b ["ic_cap"])else None )
                kayit ["onu_kapali"]=bool (np .isfinite (b ["ileri"])and float (b ["ileri"])<5.0 )
            (TP if i in cift else FP ).append (kayit )

            # FN SEBEBI: gate ONCESI havuzda candidate present miydi?
            #
            # IKI FARKLI SORU, ONCEKI SURUM KARISTIRIYORDU (2026-08-04'te yakalandi):
            #   (1) TEKIL eslemede this GT'ye candidate dustu mu?   -> urunun gercekten alabilecegi
            #   (2) Toleransinda HERHANGI a candidate present mi?    -> havuzun ham kapsamasi
            # Hafizadaki "candidate havuzu GT'nin %93.7'sini iceriyor" (2) with olculmus; benim first
            # surumum (1) with olcup %80.7 buldu and celiski like gorundu. Ikisi FARKLI SEY:
            # dense parcada birkac GT same adayi paylasir, tekil esleme birini disarida birakir.
            # Ayrimi korumak SART, because different kola bakiyorlar:
            #   ADAY_YOK    -> real uretim/cozunurluk kaybi   -> A/F4 (temsil)
            #   ADAY_KALABALIK -> candidate present but PAYLASILDI       -> cozunurluk/ayirma (F1-7)
        cift_tum ,_ =esle (Ptum ,G ,Gd ,tol )
        gt_tum ={v :a for a ,v in cift_tum .items ()}
        if len (Ptum )and len (G ):
            dd =Ptum [:,None ,:]-G [None ,:,:]
            aa =(dd *Gd [None ,:,:]).sum (-1 )
            pp =np .linalg .norm (dd -aa [...,None ]*Gd [None ,:,:],axis =-1 )
            pp =np .where (np .abs (aa )>40 ,np .inf ,pp )
            yakin_var =(pp .min (0 )<=tol )# GT basina: toleransta HERHANGI candidate
        else :
            yakin_var =np .zeros (len (G ),bool )
        for b in fn_idx :
            kayit =dict (ortak )
            if b not in gt_tum :
                kayit ["sebep"]="ADAY_YOK"if not yakin_var [b ]else "ADAY_KALABALIK"
            elif not k [gt_tum [b ]]:
                kayit ["sebep"]="GATE_REDDI"# candidate vardi, gate reddetti
            else :
                kayit ["sebep"]="ATAMA"# kabul edildi but baska GT'ye gitti
            FN .append (kayit )
        ortak ["_havuz_kapsama"]=float (yakin_var .mean ())if len (G )else 1.0 
        KAPSAMA .append ((len (G ),float (yakin_var .sum ())))

    print (f"TP {len (TP )} | FP {len (FP )} | FN {len (FN )}")
    print (f"\n{'regime':<12}{'TP':>7}{'FP':>7}{'FN':>7}{'precision':>11}{'duyarlilik':>12}")
    for rj in ("dusuk-CP","yuksek-CP"):
        t =sum (1 for x in TP if x ["regime"]==rj )
        f_ =sum (1 for x in FP if x ["regime"]==rj )
        n =sum (1 for x in FN if x ["regime"]==rj )
        print (f"{rj :<12}{t :>7}{f_ :>7}{n :>7}{t /max (t +f_ ,1 ):>11.4f}{t /max (t +n ,1 ):>12.4f}")

    print ("\nFN SEBEP DAGILIMI (hangi arm bakacak):")
    KOL ={"ADAY_YOK":"F4/A -- TEMSIL (havuzda HIC yok)",
    "ADAY_KALABALIK":"F1-7 -- COZUNURLUK (candidate var, PAYLASILDI)",
    "GATE_REDDI":"C/E -- GATE (bilgi var, karar yanlis)",
    "ATAMA":"METRIK -- coklu esleme"}
    for rj in ("dusuk-CP","yuksek-CP"):
        c =collections .Counter (x ["sebep"]for x in FN if x ["regime"]==rj )
        tp =sum (c .values ())
        print (f"  {rj } ({tp } FN):")
        for s ,n in c .most_common ():
            print (f"    {s :<12} {n :>4}  (%{100 *n /max (tp ,1 ):.1f})  -> {KOL [s ]}")

    fz =[x for x in FP if "govde_ici"in x ]
    tz =[x for x in TP if "govde_ici"in x ]
    if fz and tz :
        print (f"\nFP FIZIKSEL PROFILI ({len (fz )} FP / {len (tz )} TP, ZENGINLESME orani):")
        for ad ,fn_ in (("govde_ici",lambda x :x ["govde_ici"]),
        ("onu_kapali",lambda x :x ["onu_kapali"]),
        ("dar_agiz(<0.8mm)",lambda x :x ["ic_cap"]is not None and 0 <x ["ic_cap"]<0.8 )):
            pf =np .mean ([fn_ (x )for x in fz ]);pt =np .mean ([fn_ (x )for x in tz ])
            print (f"  {ad :<18} FP %{100 *pf :.1f} | TP %{100 *pt :.1f} | zenginlesme {pf /max (pt ,1e-9 ):.2f}x")

    ng =sum (a for a ,_ in KAPSAMA );nk =sum (b for _ ,b in KAPSAMA )
    print ("\nHAVUZ KAPSAMASI (GT'nin toleransinda HERHANGI candidate var mi):")
    print (f"  {nk :.0f}/{ng } = %{100 *nk /max (ng ,1 ):.1f}   "
    f"<- hafizadaki %93.7 ile KIYASLANABILIR olan sayi budur")
    print (f"  TEKIL eslemede alinan: {len (TP )}/{ng } = %{100 *len (TP )/max (ng ,1 ):.1f}")
    print (f"  ARADAKI FARK = kalabalik/cozunurluk kaybi")
    with io .open (CIKTI ,"w",encoding ="utf-8")as f :
        json .dump ({"TP":TP ,"FP":FP ,"FN":FN },f ,indent =1 )
    print (f"\nbanka -> {CIKTI }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
