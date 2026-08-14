# -*- coding: utf-8 -*-
"""D5-4b: GORULMEMIS URETICI SINAVI -- first times CIDDI OLCEKTE olculuyor.

[[gate-manufacturer-disi-cokusu]]: gate unseen ureticide 0.7422 -> 0.2799 cokuyordu, but that
measurement TEK manufacturer uzerindeydi. D5-4 with 514 part / 16 manufacturer present; soru first times ciddi
olcekte yanitlanabilir: **urun gordugu four ureticinin outside calisiyor mu?**

UC MODEL AYNI KUMEDE:
  DAGITILAN : canli urun gate'i (w2 with egitildi -- this ureticileri HIC gormedi)
  TABAN     : only old parts, protocol filtreli
  v3        : old + new (exam kumesi and ikizleri CIKARILMIS)

Bu cluster for UCU DE ornekelem-DISI -- i.e. numbers dogrudan kiyaslanabilir. (194'luk measurement
kumesinde boyle DEGILDI: dagitilan gate orada sample-ICI, see.
[[f2-12-ara-measurement-and-leakage-buyuklugu]].)

TEZ DEGISMEZ: no sey egitilmez; this a SINAVDIR.
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

from f2_12_data_kolu import egit ,olc 

SINAV_DER ="results/_der_sinav_yeni.pkl"
SINAV_JSON ="results/d5_4_exam_set.json"
MAKBUZ ="results/d5_4b_exam_measure.json"


def main ():
    import protocol 
    protocol .tez_dogrula ()
    import wire_gate 

    with open (SINAV_DER ,"rb")as f :
        DER =pickle .load (f )
    sv =json .load (io .open (SINAV_JSON ,encoding ="utf-8"))
    c =collections .Counter (r ["mfg"]for r in DER )
    print (f"SINAV KUMESI: {len (DER )} part / {len (c )} manufacturer (muhur {sv ['sha16']})")
    print (f"  {dict (c .most_common ())}\n")

    MOD ={}
    g =json .load (io .open ("cp_config.json",encoding ="utf-8"))["current_product"]["wire_gate"]
    with open (g ["path"],"rb")as f :
        MOD ["DAGITILAN (canli urun)"]=pickle .load (f )
    for ad ,npz in (("TABAN (only old)","results/zengin_parite_v3_taban.npz"),
    ("v3 (old + YENI)","results/zengin_parite_v3.npz")):
        m ,bilgi =egit (npz )
        MOD [ad ]=m 
        print (f"  {ad }: corpus {bilgi ['part']} part / {bilgi ['manufacturer']} manufacturer egitildi")

    S ={}
    print (f"\n{'model':<24}{'F1':>9}{'TP':>7}{'FP':>7}{'FN':>7}")
    for ad ,m in MOD .items ():
        r =olc (m ,DER )
        S [ad ]=r 
        print (f"{ad :<24}{r ['F1']:>9.4f}{r ['TP']:>7}{r ['FP']:>7}{r ['FN']:>7}")

        # URETICI KIRILIMI -- only n>=5 olanlar karar gives
    say =collections .Counter (r ["mfg"]for r in DER )
    ort =[m for m in say if say [m ]>=5 ]
    print (f"\n{'manufacturer':<8}{'part':>7}"+"".join (f"{a [:12 ]:>14}"for a in MOD ))
    for u in sorted (ort ,key =lambda x :-say [x ]):
        sat ="".join (f"{S [a ]['manufacturer'].get (u ,float ('nan')):>14.4f}"for a in MOD )
        print (f"{u :<8}{say [u ]:>7}{sat }")

    d =S ["DAGITILAN (canli urun)"]
    print (f"\nCANLI URUN, GORULMEMIS 16 URETICIDE: F1 {d ['F1']:.4f}")
    print (f"  194'luk measurement kumesinde (PXC+WEI, orneklem-ICI): 0.8536")
    print (f"  resmi headline (bekcili alt cluster): 0.7584")
    print (f"  -> aradaki diff URETICI GENELLEMESININ bedeli")
    ur =[S ["DAGITILAN (canli urun)"]["manufacturer"][u ]for u in ort ]
    print (f"  manufacturer yayilimi: en iyi {max (ur ):.4f} / en kotu {min (ur ):.4f} "
    f"(diff {max (ur )-min (ur ):.4f})")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"sinav_sha16":sv ["sha16"],"n_parca":len (DER ),
        "sonuc":{a :{k :v for k ,v in r .items ()}for a ,r in S .items ()},
        "uretici_say":dict (say )},f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
