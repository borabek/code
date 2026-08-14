# -*- coding: utf-8 -*-
"""ADAY RECALL -- "0.9372" NE DEGILDIR, ne oldugu and real ceiling.

2026-08-01 DENETIM BULGUSU: ceiling merdiveninde `4 ADAY TAVANI = 0.9372` satirini "candidates
GT'nin %93.7'sini iceriyor" diye okudum. **YANLIS.** 0.9372 a F1'dir: oracle secimle elde
edilen, regime-agirlikli F1. Bir GT for candidate VARSA TP sayilir, YOKSA FN sayilir, and FP=0
alinir -- i.e. 0.9372 = 2R/(1+R) bicimindeki a F1, RECALL DEGIL.

Bu IMPORTANT because hedefin ulasilabilirligi buna bakiyor: F1 = 2R/(1+R) oldugu for
R = 0.88 -> F1 0.936 gorunur; but real recall 0.88'dir and FP never sifir olmayacaktir.

Bu betik UC sayiyi ayri ayri gives:
    HAM recall           : matched GT / total GT (parts havuzlanmis)
    AGIRLIKLI recall     : regime agirligiyla (low-CP 0.895 / very-CP 0.105)
    REJIM recall         : low-CP and very-CP ayri
and bunlardan TESPIT F1 TAVANINI (oracle secim, FP=0) turetir.
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def eslesen_gt (P ,G ,Gd ,diag ):
    """Kac GT'nin YAKININDA at least a candidate present (detection toleransi, candidate secimi YOK)."""
    if not len (P )or not len (G ):
        return 0 
    diff =P [:,None ,:]-G [None ,:,:]
    al =(diff *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
    pe =np .where (np .abs (al )>40 ,np .inf ,pe )
    tt =max (3.0 ,0.06 *diag )
    hit =np .zeros (len (G ),bool );used =set ()
    for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
    for a in range (len (P ))for b in range (len (G ))):
        if d_ >tt or a_ in used or hit [b_ ]:
            continue 
        hit [b_ ]=True ;used .add (a_ )
    return int (hit .sum ())


def main ():
    import measure_set 
    from sina_cluster import W 

    DER ,rap =measure_set .cluster ()
    measure_set .rapor_bas (rap )

    say ={"low":[0 ,0 ],"very":[0 ,0 ]}# [matched, total]
    for r in DER :
        G =np .asarray (r ["G"],float )
        if not len (G ):
            continue 
        P =r ["P"]if r ["X"]is not None else np .zeros ((0 ,3 ))
        rj ="very"if r ["n"]>=8 else "low"
        say [rj ][0 ]+=eslesen_gt (P ,G ,np .asarray (r ["Gd"],float ),float (r ["diag"]))
        say [rj ][1 ]+=len (G )

    ham_e =sum (v [0 ]for v in say .values ());ham_t =sum (v [1 ]for v in say .values ())
    ham_R =ham_e /max (ham_t ,1 )
    rej ={k :(v [0 ]/max (v [1 ],1 ))for k ,v in say .items ()}
    agir_R =sum (W [k ]*rej [k ]for k in W )

    f1_tavan =lambda R :2 *R /(1 +R )if R >0 else 0.0 
    print (f"\n{'olcu':<26}{'value':>9}{'-> F1 tavani':>14}")
    print (f"{'HAM recall (havuzlanmis)':<26}{ham_R :>9.4f}{f1_tavan (ham_R ):>14.4f}")
    print (f"{'AGIRLIKLI recall':<26}{agir_R :>9.4f}{f1_tavan (agir_R ):>14.4f}")
    for k in ("low","very"):
        print (f"{'  '+k +'-CP recall':<26}{rej [k ]:>9.4f}{f1_tavan (rej [k ]):>14.4f}"
        f"   ({say [k ][0 ]}/{say [k ][1 ]} GT)")
        # REJIM-AGIRLIKLI F1 TAVANI: each rejimin own F1 tavani, after agirlikli mean
    ceiling =sum (W [k ]*f1_tavan (rej [k ])for k in W )
    print (f"\nREJIM-AGIRLIKLI TESPIT F1 TAVANI: {ceiling :.4f}")
    print (f"  (this, 'oracle secim + FP=0' varsayimiyla ULASILABILECEK EN YUKSEK detection F1'idir)")
    print (f"\nDIKKAT: 0.9372 RECALL DEGILDIR -- oracle secimle measured_path F1'dir. Gercek candidate")
    print (f"recall'i {agir_R :.4f} (agirlikli) / {ham_R :.4f} (ham).")
    if ceiling <0.90 :
        print (f"\n=> TESPIT 0.90 MEVCUT ADAYLARLA MATEMATIKSEL OLARAK IMKANSIZ "
        f"(ceiling {ceiling :.4f}).")
        print (f"   Darbogaz very-CP recall'i: {rej ['very']:.4f} ({say ['very'][0 ]}/{say ['very'][1 ]} GT).")
    with io .open ("results/candidate_recall.json","w",encoding ="utf-8")as f :
        json .dump ({"ham_recall":ham_R ,"agirlikli_recall":agir_R ,
        "rejim_recall":rej ,"rejim_sayilar":say ,
        "tespit_f1_tavani":ceiling },f ,indent =1 )
    print ("\nmakbuz -> results/candidate_recall.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
