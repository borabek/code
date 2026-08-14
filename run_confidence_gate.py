# -*- coding: utf-8 -*-
"""GUVEN KAPISI: robotun kullandigi isaretlerin kesinligi >= 0.90 olsun.

SAHA GERCEGI: akis two rejimin karisimi -- bilinen markadan new model + never
bilinmeyen markadan new model. Tam otomatik single number 0.90 this karisimda
VERILEMEZ (ceiling ~0.84-0.86). Verilebilecek soz sudur:

    "Robotun kullandigi isaretler >=0.90 kesinliktedir;
     this kalitede otomatik KAPSAMA this an %X'tir."

Iki number AYRI raporlanir, never single a 0.90'a karistirilmaz.

YONTEM: each tahminin skoru and correct olup olmadigi measurement makbuzunda duruyor
(`parca_kirilim[pid]["skor"] / ["correct"]`). Skor esigi t for:
    precision(t) = correct(skor>=t) / prediction(skor>=t)
    kapsama(t)  = prediction(skor>=t) / tum tahminler        (sign kapsamasi)
    GT kapsama  = correct(skor>=t) / GT                     (islenen CP orani)
En small t secilir ki precision >= TARGET olsun.

KALIBRASYON / RAPOR AYRIMI: threshold KALIBRASYON makbuzunda secilir (training ya da
gelistirme kumesi), SINAV makbuzunda only UYGULANIR. Sinavda threshold aramak,
sinavdan ayar cekmek olurdu.

Kullanim:
    python run_confidence_gate.py <kalibrasyon.json> [exam.json] [hedef]
"""
import json 
import os 
import sys 

import numpy as np 

HEDEF =0.90 


def cift (receipt ):
    """Makbuzdan (skor, correct) dizileri + GT count."""
    s =receipt ["sonuc"]
    # Pay makbuzu `parca_tp_fp_fn`, birlestirilmis receipt `parca_kirilim`
    # yaziyor. Ikisini de kabul et; single isim aramak sessizce BOS sonuc veriyordu.
    kir =s .get ("parca_kirilim")or s .get ("parca_tp_fp_fn")or {}
    S ,Y =[],[]
    n_gt =0 
    for v in kir .values ():
        sk =v .get ("skor")
        dg =v .get ("dogru")
        n_gt +=v ["rob"][0 ]+v ["rob"][2 ]# TP + FN
        if not sk or dg is None or len (sk )!=len (dg ):
            continue 
        S +=list (sk )
        Y +=list (dg )
    return np .asarray (S ,float ),np .asarray (Y ,int ),n_gt 


def egri (S ,Y ,n_gt ):
    """Azalan threshold along (threshold, precision, isaret_kapsama, gt_kapsama)."""
    if not len (S ):
        return []
    i =np .argsort (-S )
    s ,y =S [i ],Y [i ]
    dog =np .cumsum (y )
    n =np .arange (1 ,len (s )+1 )
    return list (zip (s ,dog /n ,n /len (s ),dog /max (n_gt ,1 )))


def esik_sec (S ,Y ,n_gt ,hedef ):
    """Kesinligi >= hedef tutan EN DUSUK threshold (i.e. most genis kapsama)."""
    e =egri (S ,Y ,n_gt )
    iyi =[t for t in e if t [1 ]>=hedef ]
    return iyi [-1 ]if iyi else None 


def bas (ad ,S ,Y ,n_gt ,threshold =None ,hedef =HEDEF ):
    print (f"\n--- {ad } ---")
    if not len (S ):
        print ("  skor/dogru verisi YOK (receipt eski surumle uretilmis)")
        return None 
    print (f"  tahmin {len (S )} | dogru {int (Y .sum ())} | GT {n_gt } | "
    f"ham precision {Y .mean ():.4f}")
    if threshold is None :
        r =esik_sec (S ,Y ,n_gt ,hedef )
        if r is None :
            print (f"  !! precision hicbir esikte >= {hedef :.2f} olmuyor "
            f"(en yuksek {max (t [1 ]for t in egri (S ,Y ,n_gt )):.4f})")
            return None 
        threshold =float (r [0 ])
        print (f"  SECILEN threshold {threshold :.4f} (kalibrasyon)")
    k =S >=threshold 
    if not k .any ():
        print (f"  threshold {threshold :.4f} hicbir tahmini gecirmiyor")
        return threshold 
    print (f"  threshold {threshold :.4f} -> KESINLIK {Y [k ].mean ():.4f} | "
    f"sign kapsamasi {k .mean ():.4f} | "
    f"GT kapsamasi {Y [k ].sum ()/max (n_gt ,1 ):.4f} "
    f"({int (Y [k ].sum ())}/{n_gt })")
    return threshold 


def main ():
    if len (sys .argv )<2 :
        sys .exit (__doc__ )
    kal =json .load (open (sys .argv [1 ]))
    hedef =float (sys .argv [3 ])if len (sys .argv )>3 else HEDEF 
    S ,Y ,n =cift (kal )
    threshold =bas (f"KALIBRASYON  {sys .argv [1 ]}",S ,Y ,n ,None ,hedef )

    out ={"hedef_kesinlik":hedef ,"threshold":threshold ,"kalibrasyon":sys .argv [1 ]}
    if threshold is not None and len (sys .argv )>2 and os .path .exists (sys .argv [2 ]):
        sin =json .load (open (sys .argv [2 ]))
        S2 ,Y2 ,n2 =cift (sin )
        bas (f"SINAV  {sys .argv [2 ]}  (threshold UYGULANIR, aranmaz)",S2 ,Y2 ,n2 ,
        threshold ,hedef )
        if len (S2 ):
            k =S2 >=threshold 
            out ["exam"]={
            "dosya":sys .argv [2 ],"precision":float (Y2 [k ].mean ())if k .any ()else 0.0 ,
            "isaret_kapsama":float (k .mean ()),
            "gt_kapsama":float (Y2 [k ].sum ()/max (n2 ,1 )),
            "onayli":int (k .sum ()),"oneri":int ((~k ).sum ())}
            print (f"\nMUHUR: robotun kullandigi isaretler "
            f"{out ['exam']['precision']:.1%} kesinlikte; bu kalitede "
            f"otomatik kapsama GT'nin {out ['exam']['gt_kapsama']:.1%}'i "
            f"({out ['exam']['onayli']} ONAYLI / "
            f"{out ['exam']['oneri']} ONERI).")
    json .dump (out ,open ("results/guven_kapisi.json","w"),indent =1 )
    print ("\nmakbuz -> results/guven_kapisi.json")


if __name__ =="__main__":
    main ()
