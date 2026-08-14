# -*- coding: utf-8 -*-
"""FB-2a: TEL/ALET yardimci etiketlerini uret.

DIAGNOSIS: tezin `Contact` sinifi tasarimi geregi "Kontaktierung *bzw.* Werkzeugeinschub" --
tel girisi and alet agzi TEK SINIF. Yani backbone, bizim ayirmak istedigimiz two seyi
BIRLESTIRMEK so as to egitildi. Sonra asagidaki gate'ten that ayrimi geri kazanmasini istiyoruz.
Dokuz bagimsiz mekanizmanin same cepheye carpmasinin koku budur.

BU BETIK yardimci supervizyonun ETIKETINI produces:
    a opening URETICI CP'siyle eslesiyorsa      -> TEL   (0)
    eslesmiyorsa                                  -> ALET  (1)
    opening disindaki each vertex                    -> MASKELI (-1, sinyal absent)

TEZ IHLALI YOK: this label ana 5-sinif etiketini DEGISTIRMEZ, ayri a dosyaya yazilir and
ayri a kafayi besler. Tezin sinif tanimi, softmax'i and kaybi aynen kalir.

ETIKET GURULTUSU (bilinerek kabul edilir): "eslesmeyen = alet" varsayimi, ureticinin
listelemedigi real a tel girisini alet diye ogretebilir. Bu risk MEASURED and small:
kutup-orgusu testi FP'lerin only %0.4'unun unutulmus kutup oldugunu showed
(listelenmis CP'lerde %70.1) -- i.e. eslesmeyen aciklklar gercekten tel girisi DEGIL.
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

DIZINLER =["_label_targets","_label_targets_2","_label_targets_3",
"_label_targets_recall","_label_targets_recall_hard"]
YARICAP =5.0 # mm, acikligin vertex bolgesi


def main ():
    import cad_eval 
    import diffusionnet as D_ 
    import robot_cp as RC 
    import wire_gate 
    from big_arbiter import eligible 
    from infer_step_cp import load_any ,step_to_mesh 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    E ={p :(jf ,s )for m ,p ,jf ,s in eligible ()}
    cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]

    hedef =[]
    for d in DIZINLER :
        if not os .path .isdir (d ):
            continue 
        for p in os .listdir (d ):
            yol =os .path .join (d ,p )
            if os .path .isdir (yol )and os .path .exists (os .path .join (yol ,f"{p }.obj")):
                hedef .append ((d ,p ,yol ))
    print (f"{len (hedef )} etiketli part | manufacturer JSON'u olan: "
    f"{sum (1 for _ ,p ,_ in hedef if p in E )}",flush =True )

    import trimesh 
    ok =tel_t =alet_t =0 
    t0 =time .time ()
    for k ,(d ,pid ,yol )in enumerate (hedef ,1 ):
        if k %15 ==0 :
            print (f"  {k }/{len (hedef )}  tel={tel_t } alet={alet_t }  {time .time ()-t0 :.0f}s",
            flush =True )
        if pid not in E :
            continue 
        jf ,stp =E [pid ]
        out_ =os .path .join (yol ,f"{pid }.aux.txt")
        if os .path .exists (out_ ):
            ok +=1 ;continue 
        try :
            m =trimesh .load (os .path .join (yol ,f"{pid }.obj"),process =False )
            V =np .ascontiguousarray (m .vertices ,np .float64 )
            F =np .ascontiguousarray (m .faces ,np .int64 )
            pbs =[]
            for model ,meta in models :
                _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,float ))
            cps ,_ ,_ ,_ =RC .derive_candidates (V ,F ,pbs ,stp ,cfg =cfg )
            if not cps :
                continue 
            P =np .array ([c ["point"]for c in cps ],float )
            Pd =np .array ([c ["direction"]for c in cps ],float )
            # --- manufacturer GT'sini BU mesh'in cercevesine tasi
            j =json .load (io .open (jf ,encoding ="utf-8-sig"))
            g =j .get ("ConnectionPoints")or []
            if not g :
                continue 
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in g ],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in g ],float )
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
            Vr ,_ =step_to_mesh (stp )
            R ,t_ ,_ =cad_eval .align_frames (Vr ,Vj )
            G =(G -t_ )@R ;Gd =Gd @R 
            # --- eslestir (urunun own olcutu)
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            tt =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
            eslesen =set ();ug =set ()
            for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
            for b in range (len (G ))):
                if not np .isfinite (dd )or dd >tt or a_ in eslesen or b_ in ug :
                    continue 
                eslesen .add (a_ );ug .add (b_ )
                # --- vertex etiketi
            y =np .full (len (V ),-1 ,np .int8 )
            for i in range (len (P )):
                r =np .linalg .norm (V -P [i ],axis =1 )<=YARICAP 
                if not r .any ():
                    continue 
                y [r ]=0 if i in eslesen else 1 
            tel_t +=int ((y ==0 ).sum ());alet_t +=int ((y ==1 ).sum ())
            np .savetxt (out_ ,y ,fmt ="%d")
            ok +=1 
        except Exception as e :
            print (f"    {pid }: {type (e ).__name__ }: {e }")
    print (f"\n{ok } part icin yardimci etiket yazildi")
    print (f"  TEL tepe {tel_t } | ALET tepe {alet_t } | ratio {alet_t /max (tel_t ,1 ):.2f}")
    with io .open ("results/fb2_aux_etiket.json","w",encoding ="utf-8")as f :
        json .dump ({"part":ok ,"tel_tepe":tel_t ,"alet_tepe":alet_t ,
        "yaricap_mm":YARICAP ,"dizinler":DIZINLER },f ,indent =1 )
    print ("receipt -> results/fb2_aux_etiket.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
