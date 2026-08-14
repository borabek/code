# -*- coding: utf-8 -*-
"""T3b: KACIRILAN CP'ler KAFES DUGUMLERINDE MI? (B kolunun IKINCI, ayri kapisi)

WHY SEPARATE OLCUM: T3 yapisal oznitelikleri gate'in KABUL ETTIGI candidates ten sinadi
and gecmedi. Ama that popülasyonda **FN HIC YOK** -- kacirilan CP'ler ya reddedildi ya never
candidate olmadi. Kafes TAMAMLAMA fikri full da onlari hedefliyor, therefore T3 onu OLCEMEZ.
Kendi sondamin kor noktasi; ayri kapatiliyor.

SORU: a parcada eslesmis (TP) candidates a array olusturuyorsa, KACIRILAN GT noktalari
that dizinin BOS DUGUMLERINDE mi duruyor?

  EVET whereas -> tamamlama real a FN ilaci; B kolu this haliyle kurulur.
  HAYIR whereas -> kacirmalar duzensiz; B tamamen kapanir.

TEZ DEGISMEZ: measurement; no sey egitilmez, `v_o` and candidate ureticisi aynen kalir.

GO: kacirilan GT'lerin >=%40'i lattice dugumune <=1.0mm otursun VE this ratio rastgele
konumlardan OPEN ARA high olsun (fake-lattice kontrolu: same parcada rastgele
noktalarin kafese oturma orani KARSILASTIRILIR -- lattice very sikysa each sey oturur).
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import wire_gate 
from t3_structural_tavan import _baskin_adim 

ESIK =1.0 # mm -- lattice dugumune oturma toleransi


def kafes_kalinti (pt_ ,TP ,e0 ,ort ,step_ ):
    """Noktanin array ekseni along most yakin lattice dugumune uzakligi (mm)."""
    if step_ <=1e-6 :
        return np .inf 
    t =(pt_ -ort )@e0 
    return abs (t /step_ -round (t /step_ ))*step_ 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    DER ,gate ,ek =T2 .yukle ()

    kalinti_fn ,kalinti_rast ,kalinti_tp =[],[],[]
    parca_ok =0 
    rng =np .random .RandomState (0 )
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        M =np .hstack ([r ["X"],r ["XR"]])
        if M .shape [1 ]*2 !=gate ["n_feat"]:
            continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (gate ,M ))
        P =np .asarray (r ["P"],float )[k ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if len (G )<4 or len (P )<3 :
            continue # array konusabilmek for at least 4 GT / 3 candidate
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        d =P [:,None ,:]-G [None ,:,:]
        al =(d *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        # tekil eslesme (metrikle AYNI rule)
        up ,ug =set (),set ()
        for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if dd >tol or a_ in up or b_ in ug :
                continue 
            up .add (a_ );ug .add (b_ )
        fn =[b for b in range (len (G ))if b not in ug ]
        tp =sorted (up )
        if not fn or len (tp )<3 :
            continue 
        TP =P [tp ]
        ort =TP .mean (0 )
        _ ,_ ,Vt =np .linalg .svd (TP -ort ,full_matrices =False )
        e0 =Vt [0 ]
        step_ =_baskin_adim ((TP -ort )@e0 )
        if step_ <=1e-6 :
            continue 
        parca_ok +=1 
        for b in fn :
            kalinti_fn .append (kafes_kalinti (G [b ],TP ,e0 ,ort ,step_ ))
        for a in tp :
            kalinti_tp .append (kafes_kalinti (P [a ],TP ,e0 ,ort ,step_ ))
            # SAHTE-KAFES KONTROLU: same parcanin boundary kutusunda rastgele points
        lo ,hi =G .min (0 ),G .max (0 )
        for _ in range (len (fn )):
            q =lo +rng .rand (3 )*(hi -lo )
            kalinti_rast .append (kafes_kalinti (q ,TP ,e0 ,ort ,step_ ))

    kf =np .array (kalinti_fn );kr =np .array (kalinti_rast );kt =np .array (kalinti_tp )
    print (f"dizi konusulabilen part: {parca_ok } | FN {len (kf )} | TP {len (kt )}")
    if not len (kf ):
        print ("olculecek FN absent");return 
    of =float ((kf <=ESIK ).mean ());orst =float ((kr <=ESIK ).mean ())
    ot =float ((kt <=ESIK ).mean ())
    print (f"\n{'cluster':<22}{'kafese oturan':>16}{'median kalinti':>18}")
    for ad ,v in (("KACIRILAN (FN)",kf ),("rastgele point",kr ),("eslesen (TP)",kt )):
        print (f"{ad :<22}{100 *(v <=ESIK ).mean ():>15.1f}%{np .median (v ):>17.2f}mm")
    kazanc =of -orst 
    print (f"\nFN kafese oturma %{100 *of :.1f} | rastgele %{100 *orst :.1f} | "
    f"FARK {100 *kazanc :+.1f} puan")
    gecti =of >=0.40 and kazanc >=0.15 
    print (f"GO (FN >=%40 VE rastgeleden >=15 puan yuksek) -> "
    f"{'GECTI -- lattice TAMAMLAMA real FN ilaci'if gecti else 'GECMEDI -- B TAMAMEN KAPANIR'}")
    if not gecti and of >=0.40 :
        print ("  NOT: FN orani high but rastgele de high -> lattice very sik, AYIRT ETMIYOR")
    with io .open ("results/t3b_kafes_tamamlama.json","w",encoding ="utf-8")as f :
        json .dump ({"part":parca_ok ,"n_fn":len (kf ),"esik_mm":ESIK ,
        "fn_oturan":of ,"rastgele_oturan":orst ,"tp_oturan":ot ,
        "difference":kazanc ,"gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/t3b_kafes_tamamlama.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
