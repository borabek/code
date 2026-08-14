# -*- coding: utf-8 -*-
"""DONUSUM NEREDE KAYBEDILIYOR: lateral mi, angle mi, axial mi?

Olculdu: known brand/unseen modelde pool recall 0.8697 but robot recall
0.4212 -> donusum **0.4843**. Robot tavani 0.5928, i.e. robot 0.80 IMKANSIZ.
Tavani yukseltmenin single yolu donusumu acmak.

Bu betik, TESPIT toleransinda ulasilan each GT for, that GT'ye most yakin adayin
hangi olcutu ihlal ettigini sayar:
  YANAL   dik distance > 2mm
  ACI     signed angle > 10 derece
  EKSENEL |axial| > 40mm
  IKISI   lateral VE angle
Ayrica "only lateral duzelse ne olurdu" and "only angle duzelse ne olurdu"
tavanlari verilir -- hangi cephenin kac score getirdigi however boyle bilinir.

TEZE SADIK: no turetme degistirilmiyor, only mevcut pool olculuyor.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")

YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 


def bilesenler (P ,D ,g ,gd ):
    """Aday havuzundaki HER candidate for (lateral, angle, axial) uclusu."""
    v =P -g [None ]
    n =np .linalg .norm (gd )
    if n <1e-9 :
        return None 
    u =gd /n 
    eks =v @u 
    yan =np .linalg .norm (v -eks [:,None ]*u [None ],axis =1 )
    Dn =D /(np .linalg .norm (D ,axis =1 ,keepdims =True )+1e-12 )
    aci =np .degrees (np .arccos (np .clip (Dn @u ,-1.0 ,1.0 )))# ISARETLI
    return yan ,aci ,np .abs (eks )


def main ():
    rec_ =os .environ .get ("AYR_KAYIT","results/_der_yeni.pkl")
    cluster =os .environ .get ("AYR_KUME","results/val_set.json")
    pids ={str (p )for p in json .load (open (cluster ))["pidler"]}
    R =[r for r in pickle .load (open (rec_ ,"rb"))if str (r ["pid"])in pids ]
    print (f"kayit {rec_ } | cluster {cluster } | part {len (R )}",flush =True )

    say =collections .Counter ()
    n_gt =0 
    ceiling =collections .Counter ()
    for r in R :
        G =np .asarray (r .get ("G",[]),float )
        Gd =np .asarray (r .get ("Gd",[]),float )
        if not len (G ):
            continue 
        P =np .asarray (r ["P"],float )
        D =np .asarray (r ["Pd"],float )
        dg =float (r ["diag"])
        tol =max (3.0 ,0.06 *dg )
        if not len (P ):
            n_gt +=len (G )
            say ["ADAY_YOK"]+=len (G )
            continue 
        for j in range (len (G )):
            n_gt +=1 
            b =bilesenler (P ,D ,G [j ],Gd [j ])
            if b is None :
                say ["GT_YONU_BOZUK"]+=1 
                continue 
            yan ,aci ,eks =b 
            # TESPIT toleransinda ulasilabilir mi (lateral tol, angle serbest)
            ul =(yan <=tol )&(eks <=EKSENEL )
            if not ul .any ():
                say ["ULASILAMAZ (detection tol.)"]+=1 
                continue 
            iy ,ia ,ie =yan [ul ],aci [ul ],eks [ul ]
            if ((iy <=YANAL )&(ia <=ACI )&(ie <=EKSENEL )).any ():
                say ["ROBOT TAMAM"]+=1 
                ceiling ["yalniz_yanal"]+=1 
                ceiling ["yalniz_aci"]+=1 
                continue 
            y_ok =(iy <=YANAL ).any ()
            a_ok =(ia <=ACI ).any ()
            if y_ok and a_ok :
                say ["IKISI AYRI ADAYDA"]+=1 # lateral a adayda, angle baskasinda
            elif y_ok :
                say ["ACI ihlali"]+=1 
            elif a_ok :
                say ["YANAL ihlali"]+=1 
            else :
                say ["YANAL + ACI"]+=1 
                # "only X duzelse" tavanlari
            if (ia <=ACI ).any ():
                ceiling ["yalniz_yanal"]+=1 # yanali bedava saysak
            if (iy <=YANAL ).any ():
                ceiling ["yalniz_aci"]+=1 # aciyi bedava saysak

    print (f"\nTOPLAM GT: {n_gt }")
    for k ,v in say .most_common ():
        print (f"  {k :<24} {v :>6}  %{100 *v /max (n_gt ,1 ):.1f}")
    r_tam =say ["ROBOT TAMAM"]
    print (f"\nMEVCUT robot recall      {r_tam /max (n_gt ,1 ):.4f}")
    print (f"YALNIZ YANAL duzelirse   {ceiling ['yalniz_yanal']/max (n_gt ,1 ):.4f}  "
    f"(+{(ceiling ['yalniz_yanal']-r_tam )/max (n_gt ,1 ):.4f})")
    print (f"YALNIZ ACI duzelirse     {ceiling ['yalniz_aci']/max (n_gt ,1 ):.4f}  "
    f"(+{(ceiling ['yalniz_aci']-r_tam )/max (n_gt ,1 ):.4f})")
    json .dump ({"damga":receipt_hash .damga (),"kayit":rec_ ,"cluster":cluster ,
    "n_gt":n_gt ,"sayim":dict (say ),
    "robot_recall":r_tam /max (n_gt ,1 ),
    "yalniz_yanal_tavani":ceiling ["yalniz_yanal"]/max (n_gt ,1 ),
    "yalniz_aci_tavani":ceiling ["yalniz_aci"]/max (n_gt ,1 ),
    "not":"GT basina EN IYI adayin ihlali. 'IKISI AYRI ADAYDA' = lateral "
    "a adayda, aci baskasinda saglaniyor -> tek adayda "
    "birlestirmek kazanc olurdu."},
    open (os .environ .get ("AYR_CIKTI","results/donusum_ayristir.json"),
    "w"),indent =1 )
    print ("\nmakbuz yazildi")


if __name__ =="__main__":
    main ()
