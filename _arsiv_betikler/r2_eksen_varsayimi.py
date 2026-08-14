# -*- coding: utf-8 -*-
"""R2: "BIR PARCA = BIR EKSEN" varsayimini SINA (medyanla not, DAGILIMLA).

R1'de part ici GT axis benzerliginin MEDYANI 1.000 output. Medyan a varsayimi TASIYAMAZ:
|cos| kullandigimiz for BIRBIRINE TERS bakan two order da 1.000 gives, and medyan 1.000 iken
parcanin ucte biri dik may be.

Planlanan correction (part ici axis uzlasisi: susan CP'lere guvenilir eksenden direction tasi) TAM
OLARAK this varsayima dayaniyor. Yanlissa correction correct eksenleri de BOZAR -- i.e. this measurement
kill kriterinin kendisi.

OLCULEN:
  1. Her parcada GT eksenlerinin BASKIN yone according to acilari -> parcalarin yuzde kaci "single eksenli"?
  2. Tek-eksenli olmayan parts hangileri, kac CP'li?
  3. Uzlasi uygulansaydi GT'ye according to TAVAN ne olurdu (each CP'ye parcanin baskin GT ekseni
     verilseydi kac CP 10 derece inside kalirdi)?
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def baskin_eksen (D ):
    """Isaretten bagimsiz baskin direction: sum(d d^T) matrisinin most large ozvektoru."""
    C =(D [:,:,None ]*D [:,None ,:]).sum (0 )
    ev ,evec =np .linalg .eigh (C )
    return evec [:,-1 ]


def main ():
    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    with open ("results/r0_kayit.pkl","rb")as f :
        K =pickle .load (f )

    print ("1) PARCA ICI GT EKSEN TUTARLILIGI")
    tek ,karisik ,sapma_hep =0 ,[],[]
    for r in DER :
        G =np .asarray (r ["Gd"],float )
        if len (G )<2 :
            continue 
        b =baskin_eksen (G )
        ac =np .degrees (np .arccos (np .clip (np .abs (G @b ),0 ,1 )))
        sapma_hep .extend (ac .tolist ())
        if ac .max ()<=10.0 :
            tek +=1 
        else :
            karisik .append ((r ["pid"],len (G ),float (ac .max ()),
            int ((ac >10 ).sum ())))
    n =tek +len (karisik )
    print (f"  {n } part (>=2 CP) | TEK EKSENLI (hepsi 10 derece icinde): {tek } ({tek /n :.1%})")
    print (f"  KARISIK: {len (karisik )} ({len (karisik )/n :.1%})")
    s =np .array (sapma_hep )
    print (f"  tum CP'lerin baskin eksene sapmasi: medyan {np .median (s ):.2f} | "
    f"%90 {np .percentile (s ,90 ):.2f} | %99 {np .percentile (s ,99 ):.2f} | maks {s .max ():.2f}")
    print (f"  10 derece icinde kalan CP orani: {(s <=10 ).mean ():.1%}")

    print ("\n2) KARISIK PARCALAR (en cok sapan 10)")
    for pid ,ncp ,mx ,kac in sorted (karisik ,key =lambda x :-x [2 ])[:10 ]:
        print (f"    {pid :<14} {ncp :>3} CP | maks deviation {mx :>6.1f} deg | {kac } CP disarida")

    print ("\n3) UZLASI TAVANI: her CP'ye parcanin BASKIN GT ekseni verilseydi")
    print (f"  10 derece icinde kalirdi: {(s <=10 ).mean ():.1%} (su anki aci gecisi %81.4)")
    print ("  NOT: this a TAVAN -- gercekte baskin ekseni GT'den not TAHMINLERDEN kestirecegiz.")

    print ("\n4) TAHMINLERDEN kestirilen baskin axis GT baskin eksenine ne kadar yakin?")
    fark =[]
    for r in DER :
        G =np .asarray (r ["Gd"],float );P =np .asarray (r ["Pd"],float )
        if len (G )<1 or len (P )<2 :
            continue 
        bg ,bp =baskin_eksen (G ),baskin_eksen (P )
        fark .append (float (np .degrees (np .arccos (np .clip (abs (float (bg @bp )),0 ,1 )))))
    f =np .array (fark )
    print (f"  {len (f )} part | medyan {np .median (f ):.2f} deg | %75 {np .percentile (f ,75 ):.2f} | "
    f"%90 {np .percentile (f ,90 ):.2f}")
    print (f"  10 derece icinde: {(f <=10 ).mean ():.1%}  <- uzlasinin GERCEKCI tavani")

    print ("\n5) UZLASI KIMI DUZELTIR, KIMI BOZAR? (tespitte eslesen noktalar uzerinde)")
    # Her eslesme for: mevcut angle vs parcanin TAHMIN-baskin ekseni kullanilsaydi olacak angle
    per =collections .defaultdict (list )
    for e in K :
        per [e ["pid"]].append (e )
    DERD ={r ["pid"]:r for r in DER }
    duzelen =bozulan =same_ =0 
    yeni_aci =[]
    for pid ,es in per .items ():
        r =DERD .get (pid )
        if r is None or len (r ["Pd"])<2 :
            continue 
        bp =baskin_eksen (np .asarray (r ["Pd"],float ))
        G =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
            # uzlasi ekseninin GT eksenlerine acisi (part basina single value, GT'ler already paralel)
        a_yeni =float (np .degrees (np .arccos (np .clip (np .abs (G @bp ),0 ,1 )).min ()))if len (G )else 90.0 
        for e in es :
            yeni_aci .append (a_yeni )
            eski_ok =e ["aci"]<=10.0 
            yeni_ok =a_yeni <=10.0 
            duzelen +=int (yeni_ok and not eski_ok )
            bozulan +=int (eski_ok and not yeni_ok )
            same_ +=int (eski_ok ==yeni_ok )
    tot =duzelen +bozulan +same_ 
    print (f"  {tot } eslesme | DUZELEN {duzelen } ({duzelen /max (tot ,1 ):.1%}) | "
    f"BOZULAN {bozulan } ({bozulan /max (tot ,1 ):.1%}) | degismeyen {same_ }")
    print (f"  NET: {duzelen -bozulan :+d} nokta")
    print ("  -> BOZULAN count buyukse uzlasi HERKESE not, only GUVENSIZ eksenlere "
    "uygulanmali (selector uzlasi).")

    with open ("results/r2_eksen_varsayimi.json","w",encoding ="utf-8")as f_ :
        json .dump ({"tek_eksenli_parca_orani":tek /max (n ,1 ),
        "cp_10deg_icinde":float ((s <=10 ).mean ()),
        "tahmin_baskin_10deg":float ((f <=10 ).mean ()),
        "uzlasi_duzelen":duzelen ,"uzlasi_bozulan":bozulan ,
        "uzlasi_net":duzelen -bozulan },f_ ,indent =1 )
    print ("\nmakbuz -> results/r2_eksen_varsayimi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
