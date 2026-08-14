# -*- coding: utf-8 -*-
"""B1: 188 ADAYSIZ CP'NIN TESHISI -- Rota B'nin first kapisi.

MEASURED: 1323 manufacturer CP'sinin 188'ine (%14.2) HICBIR candidate ulasmiyor. Aday havuzunun
recall tavani 0.8579. Bu 188 CP, feature-uzayi tavaninin (~0.86) DISINDAKI single aciktir:
onlari kazanmak new feature not, DAHA IYI ADAY URETIMI gerektirir.

SORU (pahali a etiketleme kararini saatler inside cozer):
  (a) Segmentasyon that konumda CableEntry/Contact DEMIYOR
      -> ontoloji/label sorunu. Hedefli label kampanyasi HAKLI.
  (b) Diyor but ADAY URETIMI eliyor (min_v, cluster, depth esikleri)
      -> UCUZ threshold duzeltmesi. Etiket kampanyasina GEREK YOK.

YONTEM: each adaysiz CP for, konumunun cevresindeki mesh vertex noktalarinda 4 modelin
mean CE+CT olasiligina bakilir. Karsilastirma for ULASILAN CP'lerde same measurement
is done -- i.e. "yeterli olasilik" esigi veriden gelir, elle konmaz.

KILL: (b) baskinsa (>%50) label kampanyasi ACILMAZ, turetme duzeltilir.
"""
import io 
import json 
import os 
import sys 
import time 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
YARICAP =4.0 


def main ():
    import diffusionnet as D_ 
    import measure_set 
    import thesis_remesh 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from infer_step_cp import load_any ,step_to_mesh 
    from gece_kilit import guard 

    guard ("b1")
    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]

    ULASAN ,ADAYSIZ =[],[]
    t0 =time .time ()
    for k ,r in enumerate (DER ,1 ):
        if k %25 ==0 :
            print (f"  {k }/{len (DER )}  {time .time ()-t0 :.0f}s",flush =True )
            guard (f"b1 {k }")
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        P =np .asarray (r ["P"],float )if r ["P"]is not None else np .zeros ((0 ,3 ))
        # hangi GT'ye candidate ULASIYOR
        if len (P ):
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            tt =max (3.0 ,0.06 *float (r ["diag"]))
            ul =pe .min (0 )<=tt 
        else :
            ul =np .zeros (len (G ),bool )
        if ul .all ():
            continue # this parcada adaysiz CP absent -> inference gereksiz
        try :
            Vr ,Fr =step_to_mesh (stp [r ["pid"]])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            pbs =[]
            for model ,meta in models :
                _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,float ))
            q =sum (pbs )/len (pbs )
        except Exception as e :
            print (f"    {r ['pid']}: {type (e ).__name__ }")
            continue 
        cect =q [:,CE ]+q [:,CT ]
        arg =q .argmax (-1 )
        for b in range (len (G )):
            d =np .linalg .norm (V -G [b ],axis =1 )
            yak =d <=YARICAP 
            if yak .sum ()<3 :
                yak =d <=(YARICAP *2 )
            if yak .sum ()<1 :
                continue 
            rec_ ={"pid":r ["pid"],"cect_max":float (cect [yak ].max ()),
            "cect_ort":float (cect [yak ].mean ()),
            "ce_ct_tepe":int (np .isin (arg [yak ],[CE ,CT ]).sum ()),
            "vertex":int (yak .sum ())}
            (ULASAN if ul [b ]else ADAYSIZ ).append (rec_ )
    print (f"\nULASAN CP {len (ULASAN )} | ADAYSIZ CP {len (ADAYSIZ )}")
    if not ADAYSIZ :
        print ("adaysiz CP bulunamadi");return 

    def oz (A ,ad ):
        cm =np .array ([x ["cect_max"]for x in A ]);ct =np .array ([x ["ce_ct_tepe"]for x in A ])
        print (f"{ad :<12}CE+CT olasilik max: medyan {np .median (cm ):.3f}  "
        f"|  CE/CT tepe sayisi: medyan {np .median (ct ):.0f}  "
        f"|  hic CE/CT tepesi YOK: {float ((ct ==0 ).mean ()):.1%}")
        return cm ,ct 
    print ()
    cmU ,ctU =oz (ULASAN ,"ULASAN")
    cmA ,ctA =oz (ADAYSIZ ,"ADAYSIZ")

    # ESIK VERIDEN: ulasan CP'lerin 10. yuzdeligi = "segmentasyon yeterince konustu" siniri
    threshold =float (np .percentile (cmU ,10 ))
    b_pay =float ((cmA >=threshold ).mean ())
    print (f"\nESIK (ulasanlarin 10. yuzdeligi): CE+CT olasilik {threshold :.3f}")
    print (f"  ADAYSIZ CP'lerin {b_pay :.1%}'inde segmentasyon ZATEN bu esigin ustunde")
    print ()
    print (f"  (a) SEGMENTASYON SUSUYOR   : {1 -b_pay :.1%}  -> ontoloji/etiket sorunu")
    print (f"  (b) SEGMENTASYON KONUSUYOR : {b_pay :.1%}  -> ADAY URETIMI eliyor (ucuz duzeltme)")
    print ()
    if b_pay >0.5 :
        print ("DECISION: (b) BASKIN -> label kampanyasi ACILMAZ. Once candidate uretimi esikleri.")
    else :
        print ("DECISION: (a) BASKIN -> segmentasyon that acikliklari GORMUYOR.")
        print ("       Hedefli label kampanyasi HAKLI: this 188 CP'nin turu ogretilmeli.")
    with io .open ("results/b1_adaysiz_teshis.json","w",encoding ="utf-8")as f :
        json .dump ({"ulasan":len (ULASAN ),"adaysiz":len (ADAYSIZ ),"threshold":threshold ,
        "b_pay_uretim":b_pay ,"a_pay_segmentasyon":1 -b_pay ,
        "adaysiz_hic_tepe_yok":float ((ctA ==0 ).mean ()),
        "ulasan_hic_tepe_yok":float ((ctU ==0 ).mean ())},f ,indent =1 )
    print ("receipt -> results/b1_adaysiz_teshis.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
