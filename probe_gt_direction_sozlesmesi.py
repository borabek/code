# -*- coding: utf-8 -*-
"""GT YON SOZLESMESI TUTARLI MI? (brand inside and markalar between)

FINDING (2026-08-12, III. KOL): analitik axis kullanildiginda SUPU/MOR'da
"merkezden agza" (DISARI) sign correct sonuc veriyor, UPUN'da whereas "agizdan
merkeze" (ICERI). Yani GT'nin direction isareti markaya according to DEGISIYOR gorunuyor.

BU CIDDI. Robot metrigi ISARETLI angle kullaniyor; sozlesme tutarsizsa correct
tahminler cezalandiriliyor may be and this D7 dahil BUTUN direction olcumlerini
etkiler.

BU SONDA dogrudan GT'ye bakar: each CP for direction, parcanin weight merkezinden
DISARI mi ICERI mi bakiyor? Olcu: (CP - centre) . direction isareti.

  disari_oran : parcadaki CP'lerin kaci DISARI bakiyor
Marka inside 0'a ya da 1'e yakinsa sozlesme TUTARLI (yonu ne olursa olsun).
0.5 civarindaysa part inside bile KARISIK demektir -- that zaman sorun
sozlesme not, GT ya da geometri.

D7'ye BAKILMAZ (only d6 + full kayitlari).
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402


def main ():
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    for pid ,r in kay .items ():
        G =np .asarray (r .get ("G",[]),float )
        Gd =np .asarray (r .get ("Gd",[]),float )
        if len (G )<2 or len (Gd )!=len (G ):
            continue 
        n =np .linalg .norm (Gd ,axis =1 ,keepdims =True )
        if (n <1e-9 ).any ():
            continue 
        Gn =Gd /n 
        center_ =G .mean (0 )
        disa =((G -center_ )*Gn ).sum (1 )>0 
        a =ist [r .get ("mfg","?")]
        a ["part"].append (1 )
        a ["gt"].append (len (G ))
        a ["disari"].append (float (disa .mean ()))
        # PARCA ICI tutarlilik: cogunluk ne up to baskin
        a ["baskinlik"].append (float (max (disa .mean (),1 -disa .mean ())))

    print (f"{'brand':<8}{'part':>7}{'GT':>7}{'DISARI orani':>14}"
    f"{'part ici baskinlik':>21}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        if len (a ["part"])<5 :
            continue 
        d =float (np .mean (a ["disari"]))
        b =float (np .mean (a ["baskinlik"]))
        out [m_ ]={"part":len (a ["part"]),"gt":sum (a ["gt"]),
        "disari_orani":d ,"parca_ici_baskinlik":b }
        print (f"{m_ :<8}{len (a ['part']):>7}{sum (a ['gt']):>7}{d :>14.3f}"
        f"{b :>21.3f}")
    json .dump ({"brand":out ,
    "not":"disari_orani = GT yonunun part merkezinden DISARI "
    "bakma orani. parca_ici_baskinlik = each parcada "
    "cogunluk yonun payi (1.0 = part icinde tam tutarli). "
    "D7'ye BAKILMADI."},
    open ("results/gt_direction_sozlesmesi.json","w"),indent =1 )
    print ("\nmakbuz -> results/gt_direction_sozlesmesi.json")
    print ("OKUMA:")
    print ("  baskinlik ~1.0 -> part ICINDE tutarli (sozlesme present)")
    print ("  disari_orani markalar between dagilmissa -> SOZLESME MARKAYA")
    print ("     GORE DEGISIYOR; sign a PARCA OZELLIGI as ogrenilebilir")
    print ("  baskinlik ~0.5 -> part inside bile karisik; sorun more deep")


if __name__ =="__main__":
    main ()
