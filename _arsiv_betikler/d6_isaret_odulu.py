# -*- coding: utf-8 -*-
"""D6-ISARET: TERS ISARET odulu NE KADAR and NEREDE?

Otopsi: tespit TP'lerinin **%28.3'u** (335/1185) `Pd . Gd < 0`, i.e. point DOGRU,
axis DOGRU, only YON TERS. Bu robot metrigini single basina oldururken tespiti
never etkilemiyor (tespit unsigned).

UC SORU, all of them same kayitlardan:
 1. Ters sign URETICI KONVANSIYONU mu (a markanin all of them ters) otherwise dagilmis mi?
 2. Isaret CEVRILSE kac TP robot-hazir olurdu (angle<=10 VE lateral<=2)? = UST ODUL
 3. MUKEMMEL sign secicisiyle robot F1 kac olurdu? = ceiling
"""
import collections 
import glob 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

KUME ="results/d6_sinav_kumesi.json"
MAKBUZ ="results/d6_isaret_odulu.json"


def main ():
    import wire_gate 
    from sina_cluster import match_greedy ,f1w 

    sv =json .load (io .open (KUME ,encoding ="utf-8"))
    PID =set (sv ["pidler"])
    import d6_record 
    kayit =d6_record .yukle (PID )
    with open ("results/wire_gate.pkl","rb")as f :
        gate =pickle .load (f )

    umf =collections .defaultdict (lambda :[0 ,0 ])# mfg -> [ters, total TP]
    ceviri_kazanci =collections .Counter ()
    R0 ,R1 ,R2 =[],[],[]# real / hepsini-cevir / MUKEMMEL sign
    for pid ,r in kayit .items ():
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        diag =r ["diag"]
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        if r .get ("X")is not None and r .get ("P")is not None and len (r ["P"]):
            M =np .asarray (r ["X"],float )
            if M .shape [1 ]*2 ==gate ["n_feat"]:
                k =wire_gate .decision_mask (wire_gate .decision_score (gate ,M ))
                if k .any ():
                    P =np .asarray (r ["P"],float )[k ];D =np .asarray (r ["Pd"],float )[k ]
                    # GERCEK robot
        tp0 ,fp0 ,fn0 ,_ =match_greedy (P ,D ,G ,Gd ,diag ,2.0 ,10.0 ,False ,signed =True )
        R0 .append ((rj ,tp0 ,fp0 ,fn0 ))
        # HEPSINI CEVIR (kontrol: konvansiyon global mi?)
        tp1 ,fp1 ,fn1 ,_ =match_greedy (P ,-D ,G ,Gd ,diag ,2.0 ,10.0 ,False ,signed =True )
        R1 .append ((rj ,tp1 ,fp1 ,fn1 ))
        # MUKEMMEL ISARET: each candidate for GT'ye most yakin sign secilseydi.
        # Isaretsiz esleme yapip, eslesen ciftte isareti GT'ye according to duzelt.
        Dk =D .copy ()
        if len (P )and len (G ):
            _t ,_f ,_n ,bi =match_greedy (P ,D ,G ,Gd ,diag ,0.0 ,180.0 ,True )
            for (pi ,gi ,_y ,_a ,_ac ,_b )in bi ["eslesme"]:
                if float (D [pi ]@Gd [gi ])<0 :
                    Dk [pi ]=-D [pi ]
        tp2 ,fp2 ,fn2 ,_ =match_greedy (P ,Dk ,G ,Gd ,diag ,2.0 ,10.0 ,False ,signed =True )
        R2 .append ((rj ,tp2 ,fp2 ,fn2 ))
        # manufacturer kirilimi + cevrilince kurtulan count
        if len (P )and len (G ):
            _t ,_f ,_n ,bi =match_greedy (P ,D ,G ,Gd ,diag ,0.0 ,180.0 ,True )
            for (pi ,gi ,yan ,_ax ,_aci ,_b )in bi ["eslesme"]:
                c =float (D [pi ]@Gd [gi ])
                umf [r ["mfg"]][1 ]+=1 
                if c <0 :
                    umf [r ["mfg"]][0 ]+=1 
                    aci_c =np .degrees (np .arccos (np .clip (-c ,-1 ,1 )))
                    if aci_c <=10.0 and yan <=2.0 :
                        ceviri_kazanci ["KURTULUR"]+=1 
                    elif aci_c >10.0 and yan <=2.0 :
                        ceviri_kazanci ["aci_hala_kotu"]+=1 
                    elif aci_c <=10.0 :
                        ceviri_kazanci ["yanal_hala_kotu"]+=1 
                    else :
                        ceviri_kazanci ["ikisi_de_kotu"]+=1 

    print ("TERS ISARET URETICI KIRILIMI (tespit TP'leri icinde):")
    print (f"{'manufacturer':<8}{'ters':>7}{'TP':>7}{'ratio':>8}")
    for m ,(t ,n )in sorted (umf .items (),key =lambda kv :-kv [1 ][1 ]):
        print (f"{m :<8}{t :>7}{n :>7}{100 *t /max (n ,1 ):>7.1f}%")
    tt =sum (v [0 ]for v in umf .values ());nn =sum (v [1 ]for v in umf .values ())
    print (f"{'TOPLAM':<8}{tt :>7}{nn :>7}{100 *tt /max (nn ,1 ):>7.1f}%")

    print (f"\nISARET CEVRILSE TERS OLANLARIN KADERI ({tt } adet):")
    for k ,v in ceviri_kazanci .most_common ():
        print (f"  {k :<16}{v :>6}  %{100 *v /max (tt ,1 ):.1f}")

    print (f"\n{'':<34}{'ROBOT F1':>10}")
    print (f"{'GERCEK':<34}{f1w (R0 ):>10.4f}")
    print (f"{'HEPSINI CEVIR (konvansiyon testi)':<34}{f1w (R1 ):>10.4f}")
    print (f"{'MUKEMMEL ISARET (ceiling)':<34}{f1w (R2 ):>10.4f}")
    json .dump ({"manufacturer":{m :v for m ,v in umf .items ()},
    "ceviri":dict (ceviri_kazanci ),"robot_gercek":f1w (R0 ),
    "robot_hepsi_cevrik":f1w (R1 ),"robot_mukemmel_isaret":f1w (R2 )},
    io .open (MAKBUZ ,"w",encoding ="utf-8"),indent =1 )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
