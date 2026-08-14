# -*- coding: utf-8 -*-
"""D6-OTOPSI: temiz sinavdaki kaybin TAM dagilimi + erisilebilir tavanlar.

0.70 plani TAHMINLE yazilamaz. Her GT for loss nedeni TEK TEK siniflanir:

  ADAY_YOK    : gate'ten ONCE bile toleransta candidate absent  -> TEMSIL kolu (seg/data)
  GATE_REDDI  : candidate vardi, gate hepsini reddetti       -> GATE kolu (v5 + threshold)
  KALABALIK   : candidate+gate gecti but baska GT'ye atandi  -> cluster/dedupe kolu
  ESLESTI     : detection TP

Tespit TP'lerinin robot kirilimi (2mm/10der/signed):
  ACI_FAIL, YANAL_FAIL, IKISI, TERS_ISARET, ROBOT_OK   -> selector/lateral kollari

TAVANLAR (same kayitlardan):
  KAHIN-GATE detection  : mukemmel gate mevcut adaylarla ne yapardi (FP=0 varsayimi)
  KAHIN-ACI robot    : detection same kalip TUM acilar <=10 olsaydi robot F1
"""
import collections 
import glob 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

import d6_record 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

KUME ="results/d6_exam_set.json"
MAKBUZ ="results/d6_autopsy.json"


def main ():
    import wire_gate 
    from sina_cluster import match_greedy ,f1w 

    sv =json .load (io .open (KUME ,encoding ="utf-8"))
    PID =set (sv ["pidler"])
    import d6_record 
    rec_ =d6_record .yukle (PID )
    with open ("results/wire_gate.pkl","rb")as f :
        gate =pickle .load (f )

    bucket =collections .Counter ()# GT duzeyi loss nedeni
    rb =collections .Counter ()# detection-TP robot kirilimi
    T ,R ,KG ,KA =[],[],[],[]# real detection/robot + oracle satirlari
    for pid ,r in rec_ .items ():
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="very"if r ["n"]>=8 else "low"
        diag =r ["diag"];tt =max (3.0 ,0.06 *diag )
        P0 =np .asarray (r ["P"],float )if r .get ("P")is not None else np .zeros ((0 ,3 ))
        D0 =np .asarray (r ["Pd"],float )if r .get ("Pd")is not None else np .zeros ((0 ,3 ))
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        if r .get ("X")is not None and len (P0 ):
            M =d6_record .x58 (r )
            if M is not None and M .shape [1 ]*2 ==gate ["n_feat"]:
                k =wire_gate .decision_mask (wire_gate .decision_score (gate ,M ))
                if k .any ():
                    P =P0 [k ];D =D0 [k ]

                    # --- GT duzeyi siniflama (detection kutusu: lateral<=tt, axial<=40)
        def kapsanan (Pq ):
            if not len (Pq )or not len (G ):
                return np .zeros (len (G ),bool )
            d =Pq [:,None ,:]-G [None ,:,:]
            al =(d *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            return pe .min (0 )<=tt 
        on =kapsanan (P0 )# gate ONCESI
        tp ,fp ,fn ,bi =match_greedy (P ,D ,G ,Gd ,diag ,0.0 ,180.0 ,True )
        es_gt ={e [1 ]for e in bi ["eslesme"]}
        for gi in range (len (G )):
            if gi in es_gt :
                bucket ["ESLESTI"]+=1 
            elif not on [gi ]:
                bucket ["ADAY_YOK"]+=1 
            elif not kapsanan (P )[gi ]:
                bucket ["GATE_REDDI"]+=1 
            else :
                bucket ["KALABALIK"]+=1 
                # --- detection-TP robot kirilimi
        for (pi ,gi ,yan ,_ax ,aci ,_b )in bi ["eslesme"]:
            c =float (D [pi ]@Gd [gi ])
            a_ok =(np .degrees (np .arccos (np .clip (c ,-1 ,1 )))<=10.0 )
            y_ok =yan <=2.0 
            if c <0 :
                rb ["TERS_ISARET"]+=1 
            elif a_ok and y_ok :
                rb ["ROBOT_OK"]+=1 
            elif not a_ok and not y_ok :
                rb ["IKISI"]+=1 
            elif not a_ok :
                rb ["ACI_FAIL"]+=1 
            else :
                rb ["YANAL_FAIL"]+=1 
        T .append ((rj ,tp ,fp ,fn ))
        tpr ,fpr ,fnr ,_ =match_greedy (P ,D ,G ,Gd ,diag ,2.0 ,10.0 ,False ,signed =True )
        R .append ((rj ,tpr ,fpr ,fnr ))
        # --- KAHIN-GATE: mevcut adaylarla mukemmel gate (FP=0, kapsanan each GT TP)
        kg =int (on .sum ())
        KG .append ((rj ,kg ,0 ,len (G )-kg ))
        # --- KAHIN-ACI: detection eslesmeleri same, angle kisiti kalkmis, lateral 2mm kalir
        ka =sum (1 for (_pi ,_gi ,yan ,_ax ,_aci ,_b )in bi ["eslesme"]if yan <=2.0 )
        KA .append ((rj ,ka ,fp ,len (G )-ka ))

    n_gt =sum (bucket .values ());n_tp =sum (rb .values ())
    print (f"TEMIZ SINAV {len (rec_ )} part, {n_gt } GT CP\n")
    print ("GT KAYIP DAGILIMI (detection kutusu):")
    for k ,v in bucket .most_common ():
        print (f"  {k :<12}{v :>6}  %{100 *v /n_gt :.1f}")
    print (f"\nTESPIT TP'LERININ ROBOT KIRILIMI ({n_tp } TP):")
    for k ,v in rb .most_common ():
        print (f"  {k :<12}{v :>6}  %{100 *v /max (n_tp ,1 ):.1f}")
    print (f"\n{'':<26}{'TESPIT':>9}{'ROBOT':>9}")
    print (f"{'GERCEK (gate+urun)':<26}{f1w (T ):>9.4f}{f1w (R ):>9.4f}")
    print (f"{'KAHIN-GATE (temsil tavani)':<26}{f1w (KG ):>9.4f}{'':>9}")
    print (f"{'KAHIN-ACI (lateral<=2 tavani)':<26}{'':>9}{f1w (KA ):>9.4f}")
    json .dump ({"bucket":dict (bucket ),"robot_kirilim":dict (rb ),
    "detection":f1w (T ),"robot":f1w (R ),
    "kahin_gate_tespit":f1w (KG ),"kahin_aci_robot":f1w (KA )},
    io .open (MAKBUZ ,"w",encoding ="utf-8"),indent =1 )
    print (f"\nmakbuz -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
