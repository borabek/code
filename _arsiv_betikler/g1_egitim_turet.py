# -*- coding: utf-8 -*-
"""G1: EGITIM KORPUSUNU YENI TOPLULUKLA TURET -- gate'in darbogazini acmak for.

K1 SONUCU: 8 uye + axis-farkindalikli havuzlama candidate havuzunun recall'ini
**0.7566 -> 0.9146** cikardi (+0.158). Yani "F1 0.87 pool tavaninin ustunde,
yapisal as imkansiz" hukmu ARTIK INVALID -- ceiling kirildi.

AMA uctan uca tespit kipirdamadi (+0.0042, noise). Sebebi K1'in KENDI SINIRI:
oradaki gate measurement kumesi inside 5-fold grup-caprazla, i.e. ~150 parcayla egitiliyordu and
4135 adayi suzmeye calisiyordu. Dagitilan gate 1709 parcayla egitiliyor.

Bu betik that esitsizligi kapatir: EGITIM korpusunu (measurement and LOCKED gruplari DISINDA) same
8 uyeli toplulukla and same axis-farkindalikli havuzlamayla turetir. Boylece gate YENI
candidate dagilimini gorur.

PARITE SARTI: this turetme with measurement turetmesi (G2) AYNI ensemble and AYNI havuzlama kuralini
kullanmak ZORUNDA. Hafizadaki ders: "two gate AYNI must be" -- dagitilan gate a times
bayatlayip toptan 0.1273 kaybettirmisti.

Cikti: results/gate_8uye.npz  (X22, XR, y, pids, mfg, votes)
"""
import io 
import json 
import os 
import sys 
import time 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CIKTI ="results/gate_8uye.npz"
ARA ="results/_g1_ara.pkl"
# PATOLOJIK PARCALAR: single a part tum kosuyu rehin alabiliyor. 2026-08-03'te
# 2502740000 (WEI, STEP 1.2 MB) 16+ dakika %100 CPU harcadi -- KILITLENME DEGIL
# (CPU ilerliyordu), geometride patoloji: mean part 3.5 sn, this 170 fold aykiri.
# `big_arbiter` da same sebeple --skip-parts bayragi carries. Atlanan part makbuza yazilir.
ATLA ={"2502740000"}


def main ():
    import cad_eval 
    import measure_set 
    import pickle 
    import robot_cp as RC 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from build_zengin_parite import _normaller ,zengin 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from infer_step_cp import load_any ,step_to_mesh 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    assert cfg .get ("robot_eksen_havuz"),"EKSEN HAVUZ BAYRAGI KAPALI -- parite bozulur"
    ESKI =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    YENI =[f"results/seg_extra/fb2_aux_s{i }.pt"for i in range (4 )
    if os .path .exists (f"results/seg_extra/fb2_aux_s{i }.pt")]
    CKS =list (ESKI )+YENI 
    print (f"TOPLULUK: {len (CKS )} uye ({len (ESKI )} eski + {len (YENI )} tel-farkindalikli)")

    gk =measure_set .geo_anahtarlari ()
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    tg ={r ["geo"]for r in DER }
    lg ={gk .get (p ,"absent:"+p )for p in rap ["locked_temiz"]}
    E =[(m ,p ,jf ,s )for m ,p ,jf ,s in eligible ()
    if gk .get (p ,"absent:"+p )not in tg and gk .get (p ,"absent:"+p )not in lg ]
    print (f"EGITIM havuzu: {len (E )} part (measurement + LOCKED gruplari CIKARILDI)")

    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in CKS ]
    import diffusionnet as D_ 

    X22 ,XR ,Y ,PID ,MFG ,VOT =[],[],[],[],[],[]
    if os .path .exists (ARA ):
        with open (ARA ,"rb")as f :
            X22 ,XR ,Y ,PID ,MFG ,VOT ,basla =pickle .load (f )
        print (f"ara kayittan devam: {len (Y )} candidate, {basla }. parcadan",flush =True )
    else :
        basla =0 
    t0 =time .time ();ok =0 
    for k ,(mfg ,pid ,jf ,sp )in enumerate (E ):
        if k <basla :
            continue 
        if k %20 ==0 and k >basla :
            hz =(k -basla )/max (time .time ()-t0 ,1 )
            kalan =(len (E )-k )/max (hz ,1e-6 )/60 
            print (f"  {k }/{len (E )} ok={ok } candidate={len (Y )}  {time .time ()-t0 :.0f}s "
            f"(kalan ~{kalan :.0f} dk)",flush =True )
            with open (ARA ,"wb")as f :
                pickle .dump ((X22 ,XR ,Y ,PID ,MFG ,VOT ,k ),f )
        if pid in ATLA :
            print (f"    {pid }: ATLANDI (patolojik -- bkz. ATLA)",flush =True )
            continue 
        try :
            j =json .load (io .open (jf ,encoding ="utf-8-sig"))
            g =j .get ("ConnectionPoints")or []
            if not g :
                continue 
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in g ],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in g ],float )
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
            Vr ,Fr =step_to_mesh (sp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            R ,t_ ,_ =cad_eval .align_frames (Vr ,Vj )
            Gm =(G -t_ )@R ;Gdm =Gd @R 
            pbs =[]
            for model ,meta in models :
                _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,float ))
            cps ,probs ,_ ,_ =RC .derive_candidates (V ,F ,pbs ,sp ,cfg =cfg )
            if not cps :
                continue 
            P =np .array ([c ["point"]for c in cps ],float )
            xa =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ,step_path =sp )
            xr =zengin (V ,F ,probs ,cps ,_normaller (V ,F ))
            diff =P [:,None ,:]-Gm [None ,:,:]
            al =(diff *Gdm [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gdm [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            tt =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
            yy =np .zeros (len (P ),int );up ,ug =set (),set ()
            for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
            for b in range (len (Gm ))):
                if not np .isfinite (d_ )or d_ >tt or a_ in up or b_ in ug :
                    continue 
                up .add (a_ );ug .add (b_ );yy [a_ ]=1 
            for i in range (len (P )):
                X22 .append (xa [i ]);XR .append (xr [i ]);Y .append (yy [i ])
                PID .append (pid );MFG .append (mfg )
                VOT .append (int (cps [i ].get ("_votes",1 )))
            ok +=1 
        except Exception as e :
            if k -basla <5 :
                print (f"    {pid }: {type (e ).__name__ }: {e }")
    np .savez_compressed (CIKTI ,X22 =np .array (X22 ,np .float32 ),XR =np .array (XR ,np .float32 ),
    y =np .array (Y ,np .int8 ),pids =np .array (PID ),mfg =np .array (MFG ),
    votes =np .array (VOT ,np .int16 ))
    print (f"\n{len (Y )} candidate / {ok } part -> {CIKTI }")
    print (f"  pozitif orani {np .mean (Y ):.1%} | oy dagilimi "
    f"{np .bincount (np .array (VOT ),minlength =9 )[1 :]}")
    with io .open (CIKTI .replace (".npz","_koken.json"),"w",encoding ="utf-8")as f :
        json .dump ({"ckpt":CKS ,"n_parca":ok ,"n_aday":len (Y ),"atlanan":sorted (ATLA ),
        "eksen_havuz":True ,
        "yanal_mm":cfg .get ("robot_eksen_havuz_yanal_mm",3.0 ),
        "postproc":cfg .get ("prediction_postproc",{})},f ,indent =1 )
    if os .path .exists (ARA ):
        os .remove (ARA )
    print ("receipt -> "+CIKTI .replace (".npz","_koken.json"))


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
